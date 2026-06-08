// =====================================================================
//  MetalNet Profiler - real-time training + inference dashboard
//  Dear ImGui + ImPlot + GLFW/OpenGL3
// ---------------------------------------------------------------------
//  Two live modes:
//    Live Training  - fast tiny CNN on 16x16 dummy data, streams
//                     loss/acc/throughput + scrolling compute heatmap
//    Live Inference - continuous forward-pass stress test, streams
//                     instantaneous latency + throughput (HFT-style)
//
//  Build/run:  ./scripts/run_gui.sh
// =====================================================================
#include "MetalNet/MetalNet.h"
#include "MetalNet/profiler/Profiler.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include <GLFW/glfw3.h>

#include <atomic>
#include <thread>
#include <mutex>
#include <vector>
#include <string>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <exception>
#include <omp.h>

using namespace MetalNet;
namespace P = MetalNet::prof;
using namespace std::chrono_literals;

static inline bool finite_f(float v){ return v==v && v<1e30f && v>-1e30f; }

// =====================================================================
//  Shared state
// =====================================================================
enum TestId { T_TRAIN=0, T_INFER, T_COUNT };
enum View { V_DASH=0, V_TRAIN, V_INFER, V_FILTERS, V_LOG, V_NTABS };
static const char* kTab[V_NTABS]={ "Dashboard","Training","Inference Stream","Filters","Log" };
static const char* kTest[T_COUNT]={ "Live Training","Live Inference" };

static constexpr int JOB_NONE=-1, HEATW=200;
static constexpr int IMG=16, CH=3, NCLS=10;            // tiny 16x16 RGB, 10 classes

static std::atomic<bool>  g_alive{true}, g_stop{false}, g_busy{false}, g_dark{true};
static std::atomic<int>   g_job{JOB_NONE}, g_running{-1}, g_view{V_DASH};
static std::atomic<float> g_progress{0.0f};

static std::atomic<int>   g_p_batch{64}, g_p_threads{0}, g_p_steps{3000};
static std::atomic<float> g_p_lr{0.0015f}, g_p_wd{0.008f};
static std::atomic<int>   g_filt_layer{0};   // which conv layer to visualize

static std::atomic<double> g_m_loss{0},g_m_thr{0},g_m_step{0},g_m_gflops{0},g_m_ram{0};
static std::atomic<double> g_m_acc{0},g_m_vloss{0},g_m_vacc{0};
static std::atomic<double> g_m_inflat{0},g_m_infthr{0};
static std::atomic<long long> g_m_istep{0},g_m_epoch{0},g_m_infn{0};

struct GuiState {
    std::mutex mtx;
    std::vector<float> loss_x,loss_y,ta_y, vx,vloss,vacc;
    std::vector<float> live_x,live_thr;                       // rolling per-step
    std::vector<float> inf_x,inf_lat,inf_thr;                 // rolling inference
    double train_step_ms=0,train_imgs=0,final_loss=0,final_acc=0,final_vloss=0,final_vacc=0;
    std::vector<P::LayerStat> layers;
    std::vector<float> heat; int heatL=0; std::vector<std::string> heat_names;
    std::vector<float> filt, filt0; int filt_k=0,filt_in=0,filt_out=0; int nconv=0;  // selected conv: current + initial weights
    std::vector<float> fmap; int fmap_oc=0,fmap_h=0,fmap_w=0;                          // live feature maps (activations)
    bool done[T_COUNT]={false,false};
    double peak_ram=0;
    std::vector<std::string> log;
};
static GuiState S;

static void roll(std::vector<float>& v,float x,int cap=900){ v.push_back(x); if((int)v.size()>cap) v.erase(v.begin()); }
static void slog(const std::string& s){ std::lock_guard<std::mutex> lk(S.mtx); S.log.push_back(s); if(S.log.size()>250) S.log.erase(S.log.begin()); }

// =====================================================================
//  Tiny fast model + dummy data (worker thread)
// =====================================================================
static Model build_net(){
    Model m;
    m << conv2d(3,16,3,1,1) << batchnorm2d(16) << leaky_relu(0.01f) << maxpool2d(2,2)   // 16->8
      << conv2d(16,32,3,1,1) << batchnorm2d(32) << leaky_relu(0.01f) << maxpool2d(2,2)  // 8->4
      << flatten() << dropout(0.3f) << dense(4*4*32, NCLS);
    return m;
}
// 10-class: coloured Gaussian blob (class-specific) + noise + label noise (NHWC, 16x16)
static void make_dataset(Tensor& X, Tensor& Y, int N, unsigned seed){
    X=Tensor(N,IMG,IMG,CH); Y=Tensor(N,NCLS); Y.fill(0);
    std::mt19937 g(seed); std::uniform_int_distribution<int> cls(0,NCLS-1);
    std::normal_distribution<float> noise(0,0.9f); std::uniform_real_distribution<float> u(0,1);
    float* xd=X.data.data(); float* yd=Y.data.data();
    auto cx=[&](int k){return 2.5f+(k%5)*3.0f;}; auto cy=[&](int k){return 4.0f+(k/5)*7.0f;};
    auto col=[&](int k,int c){float h=k/(float)NCLS;float v[3]={0.5f+0.5f*std::cos(6.2832f*h),0.5f+0.5f*std::cos(6.2832f*(h+0.33f)),0.5f+0.5f*std::cos(6.2832f*(h+0.66f))};return v[c];};
    const float amp=0.8f,sig=2.4f;
    for(int n=0;n<N;++n){ int k=cls(g); float CX=cx(k),CY=cy(k);
        for(int y=0;y<IMG;++y)for(int x=0;x<IMG;++x){ float d2=(x-CX)*(x-CX)+(y-CY)*(y-CY); float bump=amp*std::exp(-d2/(2*sig*sig));
            for(int c=0;c<CH;++c) xd[((n*IMG+y)*IMG+x)*CH+c]=noise(g)+bump*col(k,c); }
        int lbl=(u(g)<0.05f)?cls(g):k; yd[n*NCLS+lbl]=1.f; }
}
static int argmax_row(const float* p,int K){ int b=0; for(int j=1;j<K;++j) if(p[j]>p[b]) b=j; return b; }

static void run_train(Model& m){
    int B=g_p_batch.load(); int T=g_p_threads.load(); if(T<=0) T=omp_get_max_threads();
    int STEPS=g_p_steps.load(); float lr=g_p_lr.load(); float wd=g_p_wd.load();
    const int HWC=IMG*IMG*CH, NTRAIN=2048, NVAL=512;
    if(B<1) B=1; if(B>NVAL) B=NVAL;
    char bf[128]; std::snprintf(bf,sizeof(bf),"[Training] B=%d T=%d steps=%d lr=%.4f wd=%.3f (2048/512, 16x16, 10 classes)",B,T,STEPS,lr,wd); slog(bf);
    omp_set_num_threads(T);

    Tensor Xtr,Ytr,Xva,Yva; make_dataset(Xtr,Ytr,NTRAIN,1234); make_dataset(Xva,Yva,NVAL,9999);
    std::vector<int> idx(NTRAIN); for(int i=0;i<NTRAIN;++i) idx[i]=i; std::mt19937 rng(7);

    m.compile({B,IMG,IMG,CH}); m.train();
    Adam opt(lr,0.9f,0.999f,1e-8f,wd); CrossEntropyLoss crit;
    Tensor x(B,IMG,IMG,CH),y(B,NCLS); float* xb=x.data.data(); float* yb=y.data.data();
    const float* Xd=Xtr.data.data(); const float* Yd=Ytr.data.data();
    { std::lock_guard<std::mutex> lk(S.mtx); S.loss_x.clear();S.loss_y.clear();S.ta_y.clear();S.vx.clear();S.vloss.clear();S.vacc.clear();
      S.live_x.clear();S.live_thr.clear(); S.heat.clear(); S.heatL=0; S.heat_names.clear(); }

    auto& prof=P::Profiler::get();
    // gather conv layers + snapshot their INITIAL weights (the "before") at train start
    std::vector<Conv2D*> convs; for(auto* l:m.layers) if(auto c=dynamic_cast<Conv2D*>(l)) convs.push_back(c);
    std::vector<std::vector<float>> init_w(convs.size());
    for(size_t i=0;i<convs.size();++i) init_w[i].assign(convs[i]->weights.data.begin(),convs[i]->weights.data.end());
    { std::lock_guard<std::mutex> lk(S.mtx); S.nconv=(int)convs.size(); }
    int cursor=NTRAIN; double tot=0; int cnt=0; double le=0,ae=0; bool first=true; double vle=0,vae=0; bool vfirst=true;
    const int VAL_EVERY=40, VAL_BATCHES=3;

    auto gather=[&](int start){ for(int b=0;b<B;++b){ int r=idx[start+b];
        std::copy(Xd+(size_t)r*HWC,Xd+(size_t)r*HWC+HWC,xb+(size_t)b*HWC); std::copy(Yd+(size_t)r*NCLS,Yd+(size_t)r*NCLS+NCLS,yb+(size_t)b*NCLS);} };
    auto validate=[&](int step){
        m.eval(); double vl=0,va=0; int nb=std::min(NVAL/B,VAL_BATCHES); if(nb<1) nb=1;
        const float* XV=Xva.data.data(); const float* YV=Yva.data.data();
        for(int bi=0;bi<nb;++bi){ if(g_stop) break;
            for(int b=0;b<B;++b){ int r=bi*B+b; std::copy(XV+(size_t)r*HWC,XV+(size_t)r*HWC+HWC,xb+(size_t)b*HWC); std::copy(YV+(size_t)r*NCLS,YV+(size_t)r*NCLS+NCLS,yb+(size_t)b*NCLS); }
            Tensor& pv=m.forward(x); float l=crit.forward(pv,y); if(finite_f(l)) vl+=l;
            const float* pp=pv.data.data(); const float* yy=y.data.data(); int c=0; for(int b=0;b<B;++b) if(argmax_row(pp+b*NCLS,NCLS)==argmax_row(yy+b*NCLS,NCLS)) c++; va+=(double)c/B; }
        m.train(); vl/=nb; va/=nb;
        if(vfirst){ vle=vl; vae=va; vfirst=false; } else { vle=0.6*vle+0.4*vl; vae=0.6*vae+0.4*va; }
        g_m_vloss=vle; g_m_vacc=vae;
        std::lock_guard<std::mutex> lk(S.mtx); S.vx.push_back((float)step); S.vloss.push_back((float)vle); S.vacc.push_back((float)vae); S.final_vloss=vle; S.final_vacc=vae;
    };
    auto push_heat=[&](){
        int L=0; for(auto&l:prof.layers) if(l.init) L++; if(L==0) return;
        std::lock_guard<std::mutex> lk(S.mtx);
        if(S.heatL!=L){ S.heatL=L; S.heat.assign((size_t)L*HEATW,0.f); S.heat_names.clear();
            for(size_t i=0;i<prof.layers.size();++i){ auto&l=prof.layers[i]; if(l.init) S.heat_names.push_back(std::to_string(i)+":"+l.name); } }
        int j=0; for(size_t i=0;i<prof.layers.size();++i){ auto&l=prof.layers[i]; if(!l.init) continue;
            float* row=&S.heat[(size_t)j*HEATW]; std::memmove(row,row+1,(HEATW-1)*sizeof(float));
            float v=(float)(l.fwd_ms+l.bwd_ms); row[HEATW-1]=finite_f(v)?v:0.f; j++; }
    };

    { gather(0); Tensor& p=m.forward(x); Tensor dL=crit.backward(p,y); m.backward(dL); opt.step(m.nodes); } // warmup

    for(int s=0;s<STEPS;++s){
        if(g_stop){ slog("[Training] stopped"); break; }
        if(cursor+B>NTRAIN){ std::shuffle(idx.begin(),idx.end(),rng); cursor=0; ++g_m_epoch; }
        gather(cursor); cursor+=B;
        prof.begin_trace();
        auto t0=P::clk::now(); Tensor& p=m.forward(x); auto t1=P::clk::now();
        float loss=crit.forward(p,y); Tensor dL=crit.backward(p,y); m.backward(dL); auto t2=P::clk::now();
        prof.end_trace(); opt.step(m.nodes);
        if(!finite_f(loss)) loss=(float)le;
        const float* pp=p.data.data(); const float* yy=y.data.data(); int corr=0; for(int b=0;b<B;++b) if(argmax_row(pp+b*NCLS,NCLS)==argmax_row(yy+b*NCLS,NCLS)) corr++;
        double acc=(double)corr/B;
        double fwd=P::ms_between(t0,t1),bwd=P::ms_between(t1,t2),step=fwd+bwd; double thr=B/(step/1000.0); tot+=step; cnt++;
        double gf=(prof.total_fwd_flops()*3.0)/(step/1000.0)/1e9;
        prof.record_step(loss,thr,fwd,bwd);
        if(first){ le=loss; ae=acc; first=false; } else { le=0.92*le+0.08*loss; ae=0.92*ae+0.08*acc; }
        g_m_loss=le; g_m_acc=ae; g_m_thr=thr; g_m_step=step; g_m_gflops=gf; g_m_istep=s+1; g_m_ram=P::peak_ram_mb();
        push_heat();
        { std::lock_guard<std::mutex> lk(S.mtx);
          roll(S.loss_x,(float)(s+1)); roll(S.loss_y,(float)le); roll(S.ta_y,(float)ae);
          roll(S.live_x,(float)(s+1)); roll(S.live_thr,(float)thr);
          S.final_loss=le; S.final_acc=ae; S.train_step_ms=tot/cnt; S.train_imgs=B/((tot/cnt)/1000.0); S.layers=prof.layers;
          if(!convs.empty()){
            int sel=std::clamp(g_filt_layer.load(),0,(int)convs.size()-1); Conv2D* c=convs[sel];
            S.filt.assign(c->weights.data.begin(),c->weights.data.end()); S.filt0=init_w[sel];
            S.filt_k=c->kernel_size; S.filt_in=c->in_channels; S.filt_out=c->out_channels;
            const Tensor& o=c->output_buffer;     // activations -> feature maps (sample 0)
            if(o.shape.size()==4){ int OH=o.shape[1],OW=o.shape[2],OC=o.shape[3]; S.fmap_oc=OC;S.fmap_h=OH;S.fmap_w=OW;
              S.fmap.resize((size_t)OC*OH*OW); const float* od=o.data.data();
              for(int co=0;co<OC;++co)for(int yy=0;yy<OH;++yy)for(int xx=0;xx<OW;++xx) S.fmap[((size_t)co*OH+yy)*OW+xx]=od[((size_t)(yy*OW+xx))*OC+co]; }
          } }
        if(s%VAL_EVERY==0) validate(s);
        g_progress=(float)(s+1)/STEPS;
    }
    { std::lock_guard<std::mutex> lk(S.mtx); S.done[T_TRAIN]=true; S.peak_ram=P::peak_ram_mb();
      std::snprintf(bf,sizeof(bf),"[Training] done  train %.3f/%.0f%%  val %.3f/%.0f%%",S.final_loss,S.final_acc*100,S.final_vloss,S.final_vacc*100); }
    slog(bf);   // NB: slog() locks S.mtx - must be OUTSIDE the lock above (non-recursive mutex)
}

// Continuous forward-pass stress test: streams instantaneous latency + throughput.
static void run_infer(Model& m){
    int T=g_p_threads.load(); if(T<=0) T=omp_get_max_threads(); omp_set_num_threads(T);
    const int IB=8, HWC=IMG*IMG*CH; const int CHUNK=40;
    char bf[96]; std::snprintf(bf,sizeof(bf),"[Inference] continuous stream B=%d T=%d (Stop to end)",IB,T); slog(bf);
    m.compile({IB,IMG,IMG,CH}); m.eval();
    Tensor x(IB,IMG,IMG,CH); x.randomize();
    { std::lock_guard<std::mutex> lk(S.mtx); S.inf_x.clear(); S.inf_lat.clear(); S.inf_thr.clear(); }
    for(int i=0;i<30;i++) m.forward(x);   // warmup
    long long n=0;
    while(!g_stop){
        auto a=P::clk::now(); for(int k=0;k<CHUNK;k++){ if(g_stop) break; m.forward(x); } auto e=P::clk::now();
        double ms=P::ms_between(a,e); double lat=ms/CHUNK; double thr=(IB*CHUNK)/(ms/1000.0);
        if(!finite_f((float)lat)) continue;
        ++n; g_m_inflat=lat; g_m_infthr=thr; g_m_infn=n; g_m_ram=P::peak_ram_mb();
        std::lock_guard<std::mutex> lk(S.mtx); roll(S.inf_x,(float)n); roll(S.inf_lat,(float)lat); roll(S.inf_thr,(float)thr);
    }
    omp_set_num_threads(omp_get_max_threads());
    { std::lock_guard<std::mutex> lk(S.mtx); S.done[T_INFER]=true; }
    slog("[Inference] stopped");   // outside the lock (slog locks S.mtx)
}

static void worker_main(){
    omp_set_num_threads(omp_get_max_threads());
    Model m; try { m=build_net(); m.compile({64,IMG,IMG,CH}); } catch(...){ slog("ERROR: engine init failed"); }
    P::Profiler::get().start(omp_get_max_threads(),64);
    slog("Engine ready. Start Live Training or Live Inference.");
    while(g_alive){
        int job=g_job.exchange(JOB_NONE);
        if(job==JOB_NONE){ std::this_thread::sleep_for(4ms); continue; }
        g_stop=false; g_busy=true; g_running=job; g_progress=0;
        try { if(job==T_TRAIN) run_train(m); else if(job==T_INFER) run_infer(m); }
        catch(const std::exception& e){ slog(std::string("ERROR: ")+e.what()); }
        catch(...){ slog("ERROR: unknown exception"); }
        g_running=-1; g_busy=false; g_progress = g_stop?0.f:1.f;
    }
}

// =====================================================================
//  Theme
// =====================================================================
static void apply_theme(bool dark){
    ImGuiStyle& s=ImGui::GetStyle();
    s.WindowRounding=8;s.ChildRounding=8;s.FrameRounding=6;s.GrabRounding=6;s.PopupRounding=6;s.ScrollbarRounding=6;s.TabRounding=6;
    s.WindowPadding=ImVec2(16,14);s.FramePadding=ImVec2(11,7);s.ItemSpacing=ImVec2(10,9);s.ItemInnerSpacing=ImVec2(8,6);
    s.ScrollbarSize=12;s.GrabMinSize=12;s.WindowBorderSize=0;s.ChildBorderSize=1;s.FrameBorderSize=0;
    auto rgb=[](int r,int g,int b,float a=1){ return ImVec4(r/255.f,g/255.f,b/255.f,a); };
    ImVec4* c=s.Colors; const ImVec4 ac=rgb(99,102,241),ah=rgb(129,132,255);
    if(dark){ ImPlot::StyleColorsDark();
        c[ImGuiCol_WindowBg]=rgb(14,15,20);c[ImGuiCol_ChildBg]=rgb(22,24,31);c[ImGuiCol_PopupBg]=rgb(22,24,31);c[ImGuiCol_Border]=rgb(38,41,52);
        c[ImGuiCol_FrameBg]=rgb(30,33,42);c[ImGuiCol_FrameBgHovered]=rgb(40,44,56);c[ImGuiCol_FrameBgActive]=rgb(46,50,64);
        c[ImGuiCol_Button]=rgb(36,40,51);c[ImGuiCol_ButtonHovered]=ac;c[ImGuiCol_ButtonActive]=ah;
        c[ImGuiCol_Header]=rgb(36,40,51);c[ImGuiCol_HeaderHovered]=ac;c[ImGuiCol_HeaderActive]=ah;
        c[ImGuiCol_Text]=rgb(228,230,238);c[ImGuiCol_TextDisabled]=rgb(118,124,140);c[ImGuiCol_Separator]=rgb(38,41,52);
        c[ImGuiCol_ScrollbarBg]=rgb(14,15,20);c[ImGuiCol_ScrollbarGrab]=rgb(46,50,64);
    } else { ImPlot::StyleColorsLight();
        c[ImGuiCol_WindowBg]=rgb(236,238,243);c[ImGuiCol_ChildBg]=rgb(250,251,253);c[ImGuiCol_PopupBg]=rgb(250,251,253);c[ImGuiCol_Border]=rgb(214,218,226);
        c[ImGuiCol_FrameBg]=rgb(232,235,240);c[ImGuiCol_FrameBgHovered]=rgb(222,226,234);c[ImGuiCol_FrameBgActive]=rgb(214,219,229);
        c[ImGuiCol_Button]=rgb(226,229,236);c[ImGuiCol_ButtonHovered]=ac;c[ImGuiCol_ButtonActive]=ah;
        c[ImGuiCol_Header]=rgb(226,229,236);c[ImGuiCol_HeaderHovered]=ac;c[ImGuiCol_HeaderActive]=ah;
        c[ImGuiCol_Text]=rgb(28,32,40);c[ImGuiCol_TextDisabled]=rgb(140,146,158);c[ImGuiCol_Separator]=rgb(214,218,226);
        c[ImGuiCol_ScrollbarBg]=rgb(236,238,243);c[ImGuiCol_ScrollbarGrab]=rgb(200,205,214);
    }
    c[ImGuiCol_CheckMark]=ah;c[ImGuiCol_SliderGrab]=ac;c[ImGuiCol_SliderGrabActive]=ah;
}
static ImVec4 ACC(){ return ImVec4(0.46f,0.47f,0.96f,1); }
static ImVec4 FWD(){ return ImVec4(0.30f,0.70f,1.00f,1); }
static ImVec4 BWD(){ return ImVec4(1.00f,0.55f,0.30f,1); }
static ImVec4 GOOD(){ return ImVec4(0.25f,0.82f,0.45f,1); }
static ImVec4 MUTE(){ return g_dark.load()?ImVec4(0.55f,0.58f,0.66f,1):ImVec4(0.42f,0.46f,0.55f,1); }

// =====================================================================
//  UI helpers
// =====================================================================
static void big(const char* t,ImVec4 col,float sc=1.7f){ ImGui::PushStyleColor(ImGuiCol_Text,col); ImGui::SetWindowFontScale(sc); ImGui::TextUnformatted(t); ImGui::SetWindowFontScale(1.0f); ImGui::PopStyleColor(); }
static void card(const char* label,const char* val,ImVec4 col,float w,float h=80){
    ImGui::BeginChild(label,ImVec2(w,h),true); ImGui::TextColored(MUTE(),"%s",label); ImGui::Dummy(ImVec2(0,3)); big(val,col,1.5f); ImGui::EndChild();
}
// scrolling fixed-window line with padded Y so the trace sits mid-plot (never hugs ceiling)
static void ticker(const char* id,const std::vector<float>& x,const std::vector<float>& y,ImVec4 c,float fill,float h,int window=180){
    if(ImPlot::BeginPlot(id,ImVec2(-1,h))){
        ImPlot::SetupAxes(nullptr,nullptr,ImPlotAxisFlags_NoTickLabels,0);
        if(!x.empty()){
            double xmax=x.back(), xmin=std::max(0.0,xmax-window);
            ImPlot::SetupAxisLimits(ImAxis_X1,xmin,xmax+1,ImPlotCond_Always);
            double lo=1e30,hi=-1e30; for(size_t i=0;i<x.size();++i) if(x[i]>=xmin){ lo=std::min(lo,(double)y[i]); hi=std::max(hi,(double)y[i]); }
            if(hi<lo){ lo=0; hi=1; } double r=hi-lo; if(r<1e-6) r=(std::fabs(hi)*0.1+1);
            ImPlot::SetupAxisLimits(ImAxis_Y1, lo-0.30*r, hi+0.40*r, ImPlotCond_Always);
            if(fill>0){ ImPlot::SetNextFillStyle(c,fill); ImPlot::PlotShaded(id,x.data(),y.data(),(int)x.size(),lo-0.30*r); }
            ImPlot::SetNextLineStyle(c,2.2f); ImPlot::PlotLine(id,x.data(),y.data(),(int)x.size());
        }
        ImPlot::EndPlot();
    }
}

struct Snap {
    std::vector<float> loss_x,loss_y,ta_y,vx,vloss,vacc,live_x,live_thr,inf_x,inf_lat,inf_thr;
    double train_step_ms,train_imgs,final_loss,peak_ram,final_acc,final_vloss,final_vacc;
    std::vector<P::LayerStat> layers; std::vector<float> heat; int heatL; std::vector<std::string> heat_names;
    std::vector<float> filt,filt0; int filt_k,filt_in,filt_out,nconv;
    std::vector<float> fmap; int fmap_oc,fmap_h,fmap_w;
    bool done[T_COUNT]; std::vector<std::string> log;
};
static Snap snap(int view){
    std::lock_guard<std::mutex> lk(S.mtx); Snap d;
    d.loss_x=S.loss_x;d.loss_y=S.loss_y;d.ta_y=S.ta_y;d.vx=S.vx;d.vloss=S.vloss;d.vacc=S.vacc;
    d.live_x=S.live_x;d.live_thr=S.live_thr;d.inf_x=S.inf_x;d.inf_lat=S.inf_lat;d.inf_thr=S.inf_thr;
    d.train_step_ms=S.train_step_ms;d.train_imgs=S.train_imgs;d.final_loss=S.final_loss;d.peak_ram=S.peak_ram;
    d.final_acc=S.final_acc;d.final_vloss=S.final_vloss;d.final_vacc=S.final_vacc;
    for(int i=0;i<T_COUNT;++i) d.done[i]=S.done[i];
    if(view==V_DASH){ d.layers=S.layers; d.heat=S.heat; d.heatL=S.heatL; d.heat_names=S.heat_names; } else d.heatL=0;
    d.nconv=S.nconv;
    if(view==V_FILTERS){ d.filt=S.filt; d.filt0=S.filt0; d.filt_k=S.filt_k; d.filt_in=S.filt_in; d.filt_out=S.filt_out;
        d.fmap=S.fmap; d.fmap_oc=S.fmap_oc; d.fmap_h=S.fmap_h; d.fmap_w=S.fmap_w; }
    else { d.filt_k=d.filt_in=d.filt_out=0; d.fmap_oc=0; }
    if(view==V_LOG) d.log=S.log;
    return d;
}

// =====================================================================
//  Views
// =====================================================================
static void heatmap(const Snap& d,float h){
    if(d.heatL<=0||d.heat.empty()){ ImGui::TextDisabled("compute heatmap appears during training"); return; }
    float vmax=0.001f; for(float v:d.heat) vmax=std::max(vmax,v);
    ImPlot::PushColormap(ImPlotColormap_Plasma);
    float w=ImGui::GetContentRegionAvail().x;
    if(ImPlot::BeginPlot("Compute Heatmap  (per-layer ms, scrolling)",ImVec2(w-92,h))){
        ImPlot::SetupAxes("recent steps",nullptr,ImPlotAxisFlags_NoGridLines|ImPlotAxisFlags_NoTickLabels,ImPlotAxisFlags_NoGridLines);
        std::vector<double> yp(d.heatL); std::vector<const char*> yl(d.heatL);
        for(int i=0;i<d.heatL;++i){ yp[i]=d.heatL-0.5-i; yl[i]=d.heat_names[i].c_str(); }
        ImPlot::SetupAxisTicks(ImAxis_Y1,yp.data(),d.heatL,yl.data());
        ImPlot::PlotHeatmap("ms",d.heat.data(),d.heatL,HEATW,0.0,vmax,nullptr,ImPlotPoint(0,0),ImPlotPoint(HEATW,d.heatL));
        ImPlot::EndPlot();
    }
    ImGui::SameLine(); ImPlot::ColormapScale("ms",0.0,vmax,ImVec2(80,h)); ImPlot::PopColormap();
}
static void view_dash(const Snap& d){
    big("Live Training Dashboard",ACC(),1.9f); ImGui::TextColored(MUTE(),"Fast tiny CNN on 16x16 dummy data - streams every step");
    ImGui::Dummy(ImVec2(0,8));
    float w=(ImGui::GetContentRegionAvail().x-50)/5; char b[40];
    std::snprintf(b,sizeof(b),"%lld",g_m_istep.load());      card("STEP",b,ACC(),w); ImGui::SameLine();
    std::snprintf(b,sizeof(b),"%.3f",g_m_loss.load());       card("TRAIN LOSS",b,GOOD(),w); ImGui::SameLine();
    std::snprintf(b,sizeof(b),"%.0f%%",g_m_vacc.load()*100); card("VAL ACC",b,FWD(),w); ImGui::SameLine();
    std::snprintf(b,sizeof(b),"%.0f",g_m_gflops.load());     card("GFLOP/S",b,BWD(),w); ImGui::SameLine();
    std::snprintf(b,sizeof(b),"%.0f",g_m_thr.load());        card("IMG/S",b,ACC(),w);
    ImGui::Dummy(ImVec2(0,8));
    float rest=ImGui::GetContentRegionAvail().y; float hh=rest*0.48f-6;
    heatmap(d,hh);
    ImGui::Dummy(ImVec2(0,6));
    float half=ImGui::GetContentRegionAvail().x*0.5f-6; float lh=ImGui::GetContentRegionAvail().y;
    ImGui::BeginChild("dl",ImVec2(half,lh),false);
    if(ImPlot::BeginPlot("Loss (train vs val)",ImVec2(-1,-1))){ ImPlot::SetupAxes("step","loss",ImPlotAxisFlags_AutoFit,ImPlotAxisFlags_AutoFit);
        if(!d.loss_x.empty()){ ImPlot::SetNextLineStyle(GOOD(),2.0f); ImPlot::PlotLine("train",d.loss_x.data(),d.loss_y.data(),(int)d.loss_x.size()); }
        if(!d.vx.empty()){ ImPlot::SetNextLineStyle(BWD(),2.0f); ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle,3); ImPlot::PlotLine("val",d.vx.data(),d.vloss.data(),(int)d.vx.size()); }
        ImPlot::EndPlot(); }
    ImGui::EndChild(); ImGui::SameLine();
    ImGui::BeginChild("dt",ImVec2(-1,lh),false);
    ticker("Throughput (img/s)",d.live_x,d.live_thr,FWD(),0.18f,-1);
    ImGui::EndChild();
}
static void view_train(const Snap& d){ big("Training Detail",ACC()); ImGui::TextColored(MUTE(),"Val tracking train + plateauing above 0 = real learning, not memorisation"); ImGui::Dummy(ImVec2(0,6));
    float w=(ImGui::GetContentRegionAvail().x-50)/5; char b[32];
    std::snprintf(b,sizeof(b),"%.3f",d.final_loss); card("TRAIN LOSS",b,GOOD(),w); ImGui::SameLine();
    std::snprintf(b,sizeof(b),"%.3f",d.final_vloss);card("VAL LOSS",b,BWD(),w); ImGui::SameLine();
    std::snprintf(b,sizeof(b),"%.0f%%",d.final_acc*100); card("TRAIN ACC",b,FWD(),w); ImGui::SameLine();
    std::snprintf(b,sizeof(b),"%.0f%%",d.final_vacc*100);card("VAL ACC",b,ACC(),w); ImGui::SameLine();
    std::snprintf(b,sizeof(b),"%.2f ms",d.train_step_ms); card("STEP",b,MUTE(),w); ImGui::Dummy(ImVec2(0,6));
    float half=ImGui::GetContentRegionAvail().x*0.5f-6;
    if(ImPlot::BeginPlot("Loss",ImVec2(half,-1))){ ImPlot::SetupAxes("step","loss",ImPlotAxisFlags_AutoFit,ImPlotAxisFlags_AutoFit);
        if(!d.loss_x.empty()){ ImPlot::SetNextLineStyle(GOOD(),2.0f); ImPlot::PlotLine("train",d.loss_x.data(),d.loss_y.data(),(int)d.loss_x.size()); }
        if(!d.vx.empty()){ ImPlot::SetNextLineStyle(BWD(),2.0f); ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle,3); ImPlot::PlotLine("validation",d.vx.data(),d.vloss.data(),(int)d.vx.size()); } ImPlot::EndPlot(); }
    ImGui::SameLine();
    if(ImPlot::BeginPlot("Accuracy",ImVec2(-1,-1))){ ImPlot::SetupAxes("step","accuracy",ImPlotAxisFlags_AutoFit,ImPlotAxisFlags_AutoFit); ImPlot::SetupAxisLimits(ImAxis_Y1,0,1,ImPlotCond_Always);
        if(!d.loss_x.empty()){ ImPlot::SetNextLineStyle(FWD(),2.0f); ImPlot::PlotLine("train",d.loss_x.data(),d.ta_y.data(),(int)d.loss_x.size()); }
        if(!d.vx.empty()){ ImPlot::SetNextLineStyle(ACC(),2.0f); ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle,3); ImPlot::PlotLine("validation",d.vx.data(),d.vacc.data(),(int)d.vx.size()); } ImPlot::EndPlot(); } }
static void view_infer(const Snap& d){
    big("Live Inference Stream",ACC(),1.9f); ImGui::TextColored(MUTE(),"Continuous forward-pass stress test - instantaneous latency & throughput");
    ImGui::Dummy(ImVec2(0,8));
    float w=(ImGui::GetContentRegionAvail().x-30)/3; char b[40];
    std::snprintf(b,sizeof(b),"%.4f ms",g_m_inflat.load()); card("LATENCY / fwd",b,FWD(),w); ImGui::SameLine();
    std::snprintf(b,sizeof(b),"%.0f",g_m_infthr.load());    card("THROUGHPUT img/s",b,GOOD(),w); ImGui::SameLine();
    std::snprintf(b,sizeof(b),"%lld",g_m_infn.load());      card("BATCHES",b,ACC(),w);
    ImGui::Dummy(ImVec2(0,8));
    float lh=ImGui::GetContentRegionAvail().y; float hh=lh*0.5f-6;
    ImGui::BeginChild("ia",ImVec2(0,hh),false); ticker("Latency (ms / forward)",d.inf_x,d.inf_lat,BWD(),0.16f,-1,200); ImGui::EndChild();
    ImGui::BeginChild("ib",ImVec2(0,0),false);  ticker("Throughput (img/s)",d.inf_x,d.inf_thr,FWD(),0.16f,-1,200); ImGui::EndChild();
    if(d.inf_x.empty() && g_running.load()!=T_INFER) ImGui::TextDisabled("press \"Live Inference\" on the left to start the stream");
}
static void draw_filters(const std::vector<float>& filt,int k,int in,int out,float FS,bool labels){
    if(filt.empty()||out<=0||k<=0){ ImGui::TextDisabled("(empty)"); ImGui::Dummy(ImVec2(0,FS)); return; }
    float mx=1e-6f; for(float v:filt) mx=std::max(mx,std::fabs(v));
    auto Wt=[&](int ky,int kx,int ci,int f){ return filt[(size_t)((ky*k+kx)*in+ci)*out+f]; };
    const int COLS=8; const float GAP=18.f; float cell=FS/k; int rows=(out+COLS-1)/COLS;
    ImDrawList* dl=ImGui::GetWindowDrawList(); ImVec2 org=ImGui::GetCursorScreenPos(); ImU32 brd=ImGui::ColorConvertFloat4ToU32(MUTE());
    float rowh=FS+GAP+(labels?16.f:6.f);
    for(int f=0;f<out;++f){ int cc=f%COLS,rr=f/COLS; float fx=org.x+cc*(FS+GAP),fy=org.y+rr*rowh;
        for(int ky=0;ky<k;++ky)for(int kx=0;kx<k;++kx){ float r,g,b;
            if(in==3){ r=std::clamp(0.5f+0.5f*Wt(ky,kx,0,f)/mx,0.f,1.f); g=std::clamp(0.5f+0.5f*Wt(ky,kx,1,f)/mx,0.f,1.f); b=std::clamp(0.5f+0.5f*Wt(ky,kx,2,f)/mx,0.f,1.f); }
            else { float s=0; for(int ci=0;ci<in;++ci){ float w=Wt(ky,kx,ci,f); s+=w*w; } float gm=std::clamp(std::sqrt(s)/mx,0.f,1.f); r=g=b=gm; }
            float cx=fx+kx*cell,cy=fy+ky*cell; dl->AddRectFilled(ImVec2(cx,cy),ImVec2(cx+cell+1,cy+cell+1),ImGui::ColorConvertFloat4ToU32(ImVec4(r,g,b,1))); }
        dl->AddRect(ImVec2(fx,fy),ImVec2(fx+FS,fy+FS),brd,3.f);
        if(labels){ char l[12]; std::snprintf(l,sizeof(l),"#%d",f); dl->AddText(ImVec2(fx+2,fy+FS+1),brd,l); }
    }
    ImGui::Dummy(ImVec2(0,(float)rows*rowh+6));
}
static void draw_fmaps(const std::vector<float>& fm,int oc,int oh,int ow,float TS){
    if(fm.empty()||oc<=0){ ImGui::TextDisabled("(empty)"); ImGui::Dummy(ImVec2(0,TS)); return; }
    float lo=1e30f,hi=-1e30f; for(float v:fm){ lo=std::min(lo,v); hi=std::max(hi,v); } float rng=hi-lo; if(rng<1e-6f) rng=1;
    const int COLS=8; const float GAP=12.f; float cx=TS/ow, cy=TS/oh; int rows=(oc+COLS-1)/COLS;
    ImDrawList* dl=ImGui::GetWindowDrawList(); ImVec2 org=ImGui::GetCursorScreenPos(); ImU32 brd=ImGui::ColorConvertFloat4ToU32(MUTE());
    for(int co=0;co<oc;++co){ int cc=co%COLS,rr=co/COLS; float fx=org.x+cc*(TS+GAP),fy=org.y+rr*(TS+GAP);
        for(int y=0;y<oh;++y)for(int x=0;x<ow;++x){ float t=(fm[((size_t)co*oh+y)*ow+x]-lo)/rng;
            ImVec4 c=ImPlot::SampleColormap(std::clamp(t,0.f,1.f),ImPlotColormap_Plasma);
            float px=fx+x*cx,py=fy+y*cy; dl->AddRectFilled(ImVec2(px,py),ImVec2(px+cx+1,py+cy+1),ImGui::ColorConvertFloat4ToU32(c)); }
        dl->AddRect(ImVec2(fx,fy),ImVec2(fx+TS,fy+TS),brd,3.f); }
    ImGui::Dummy(ImVec2(0,(float)rows*(TS+GAP)+6));
}
static void view_filters(const Snap& d){
    big("Conv Filters & Feature Maps",ACC(),1.9f);
    ImGui::TextColored(MUTE(),"Layer 0 has 3 input channels -> kernels show as RGB. Deeper layers (many channels) show as grayscale magnitude.");
    ImGui::Dummy(ImVec2(0,6));
    if(d.nconv>1){ int sel=g_filt_layer.load(); ImGui::SetNextItemWidth(200);
        if(ImGui::SliderInt("conv layer",&sel,0,d.nconv-1)) g_filt_layer=sel; }
    ImGui::Dummy(ImVec2(0,4));
    if(d.filt.empty()||d.filt_out<=0){ ImGui::TextDisabled("start Live Training to populate"); return; }
    int k=d.filt_k,in=d.filt_in,out=d.filt_out;

    big("Live  (training now)",GOOD(),1.15f);
    ImGui::TextColored(MUTE(),"%d filters - %dx%d kernel - %d input channel(s)",out,k,k,in);
    draw_filters(d.filt,k,in,out,96.f,true);
    ImGui::Separator();
    big("Before  (initial random weights)",BWD(),1.15f);
    draw_filters(d.filt0,k,in,out,96.f,false);
    ImGui::Separator();
    big("Live Feature Maps  (activations on a sample image)",FWD(),1.15f);
    ImGui::TextColored(MUTE(),"what each filter \"sees\" - %d maps of %dx%d, updating every step",d.fmap_oc,d.fmap_h,d.fmap_w);
    draw_fmaps(d.fmap,d.fmap_oc,d.fmap_h,d.fmap_w,72.f);
}
static void view_log(const Snap& d){ big("Log",ACC()); ImGui::Dummy(ImVec2(0,6)); ImGui::BeginChild("lc",ImVec2(0,0),true);
    for(auto& l:d.log) ImGui::TextWrapped("%s",l.c_str()); if(!d.log.empty()) ImGui::SetScrollHereY(1.0f); ImGui::EndChild(); }

// =====================================================================
//  Header / tabs / sidebar
// =====================================================================
static void header(){
    ImGui::BeginChild("hdr",ImVec2(0,56),false);
    ImGui::SetCursorPosY(14); big("\xE2\x9A\xA1 MetalNet",ACC(),1.6f);
    ImGui::SameLine(); ImGui::SetCursorPosY(22); ImGui::TextColored(MUTE(),"  Engine Profiler");
    bool busy=g_busy.load(); float rx=ImGui::GetWindowWidth();
    ImGui::SameLine(rx-270); ImGui::SetCursorPosY(20); ImGui::TextColored(busy?GOOD():MUTE(),"%s",busy?"\xE2\x97\x8F running":"\xE2\x97\x8F idle");
    ImGui::SameLine(rx-130); ImGui::SetCursorPosY(12); bool dark=g_dark.load();
    if(ImGui::Button(dark?"  Light Mode  ":"  Dark Mode  ",ImVec2(115,32))) g_dark=!dark;
    ImGui::EndChild(); ImGui::Separator();
}
static void tabs(){ ImGui::BeginChild("tabs",ImVec2(0,42),false);
    for(int i=0;i<V_NTABS;++i){ bool a=(g_view.load()==i); if(a) ImGui::PushStyleColor(ImGuiCol_Button,ACC());
        if(ImGui::Button(kTab[i],ImVec2(0,30))) g_view=i; if(a) ImGui::PopStyleColor(); if(i<V_NTABS-1) ImGui::SameLine(); }
    ImGui::EndChild(); }
static void sidebar(const Snap& d){
    big("Parameters",ACC(),1.2f); ImGui::Dummy(ImVec2(0,4));
    bool busy=g_busy.load();
    ImGui::BeginDisabled(busy);
    ImGui::TextColored(MUTE(),"Batch Size"); int b=g_p_batch.load(); ImGui::SetNextItemWidth(-70);
    if(ImGui::InputInt("##b",&b,16,64)){ b=std::clamp(b,1,512); g_p_batch=b; }
    ImGui::TextColored(MUTE(),"Threads"); int t=g_p_threads.load(); if(t<=0)t=omp_get_max_threads(); ImGui::SetNextItemWidth(-FLT_MIN);
    if(ImGui::SliderInt("##t",&t,1,omp_get_max_threads())) g_p_threads=t;
    ImGui::TextColored(MUTE(),"Steps"); int st=g_p_steps.load(); ImGui::SetNextItemWidth(-FLT_MIN);
    if(ImGui::SliderInt("##s",&st,300,8000)) g_p_steps=st;
    ImGui::TextColored(MUTE(),"Learning Rate"); float lr=g_p_lr.load(); ImGui::SetNextItemWidth(-FLT_MIN);
    if(ImGui::SliderFloat("##lr",&lr,0.0001f,0.01f,"%.4f")) g_p_lr=lr;
    ImGui::TextColored(MUTE(),"Weight Decay"); float wd=g_p_wd.load(); ImGui::SetNextItemWidth(-FLT_MIN);
    if(ImGui::SliderFloat("##wd",&wd,0.0f,0.05f,"%.3f")) g_p_wd=wd;
    ImGui::EndDisabled();

    ImGui::Dummy(ImVec2(0,8)); ImGui::Separator(); ImGui::TextColored(MUTE(),"SIMULATIONS"); ImGui::Dummy(ImVec2(0,2));
    int run=g_running.load();
    ImGui::BeginDisabled(busy);
    for(int i=0;i<T_COUNT;++i){ bool ir=(run==i); if(ir) ImGui::PushStyleColor(ImGuiCol_Button,ACC());
        char lbl[64]; std::snprintf(lbl,sizeof(lbl),"%s%s",d.done[i]?"* ":"   ",kTest[i]);
        if(ImGui::Button(lbl,ImVec2(-FLT_MIN,40))){ g_job=i; g_view=(i==T_TRAIN?V_DASH:V_INFER); }
        if(ir) ImGui::PopStyleColor(); }
    ImGui::EndDisabled();
    ImGui::Dummy(ImVec2(0,4));
    ImGui::BeginDisabled(!busy); ImGui::PushStyleColor(ImGuiCol_Button,ImVec4(0.65f,0.20f,0.22f,1));
    if(ImGui::Button("\xE2\x96\xA0  Stop",ImVec2(-FLT_MIN,36))){ g_stop=true; slog("Stop requested"); }
    ImGui::PopStyleColor(); ImGui::EndDisabled();

    ImGui::Dummy(ImVec2(0,6)); char pb[48];
    if(run==T_TRAIN) std::snprintf(pb,sizeof(pb),"%lld / %d  (epoch %lld)",g_m_istep.load(),g_p_steps.load(),g_m_epoch.load());
    else if(run==T_INFER) std::snprintf(pb,sizeof(pb),"streaming  (%lld)",g_m_infn.load());
    else std::snprintf(pb,sizeof(pb),"idle");
    float pf = (run==T_INFER)?1.0f:std::clamp(g_progress.load(),0.f,1.f);
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram,ImVec4(0.95f,0.62f,0.10f,1));
    ImGui::ProgressBar(pf,ImVec2(-FLT_MIN,22),pb); ImGui::PopStyleColor();
    if(ImGui::Button("Export Perfetto Trace",ImVec2(-FLT_MIN,28))){ P::Profiler::get().export_chrome_trace("metalnet_trace.json"); slog("Trace -> metalnet_trace.json"); }

    ImGui::Dummy(ImVec2(0,8)); ImGui::Separator(); ImGui::TextColored(MUTE(),"LIVE METRICS"); ImGui::Dummy(ImVec2(0,4));
    auto kv=[&](const char* k,const char* f,double v,ImVec4 c){ ImGui::TextColored(MUTE(),"%s",k); ImGui::SameLine(150); char x[40]; std::snprintf(x,sizeof(x),f,v); ImGui::TextColored(c,"%s",x); };
    kv("Train loss","%.3f",g_m_loss.load(),GOOD()); kv("Train acc","%.0f%%",g_m_acc.load()*100,FWD());
    kv("Val loss","%.3f",g_m_vloss.load(),BWD()); kv("Val acc","%.0f%%",g_m_vacc.load()*100,ACC());
    kv("Infer lat","%.3f ms",g_m_inflat.load(),BWD()); kv("Infer thr","%.0f",g_m_infthr.load(),FWD());
    kv("GFLOP/s","%.0f",g_m_gflops.load(),BWD()); kv("Peak RAM","%.0f MB",g_m_ram.load(),MUTE());
}

// =====================================================================
int main(){
    if(!glfwInit()){ std::fprintf(stderr,"glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3); glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,2);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT,GLFW_TRUE);
#endif
    GLFWwindow* win=glfwCreateWindow(1620,1000,"MetalNet Profiler",nullptr,nullptr);
    if(!win){ glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win); glfwSwapInterval(1);
    IMGUI_CHECKVERSION(); ImGui::CreateContext(); ImPlot::CreateContext();
    bool cur=true; apply_theme(cur);
    ImGui_ImplGlfw_InitForOpenGL(win,true); ImGui_ImplOpenGL3_Init("#version 150");

    std::thread worker(worker_main);

    while(!glfwWindowShouldClose(win)){
        glfwPollEvents();
        if(g_dark.load()!=cur){ cur=g_dark.load(); apply_theme(cur); }
        ImGui_ImplOpenGL3_NewFrame(); ImGui_ImplGlfw_NewFrame(); ImGui::NewFrame();
        int view=g_view.load(); Snap d=snap(view);
        const ImGuiViewport* vp=ImGui::GetMainViewport(); ImGui::SetNextWindowPos(vp->WorkPos); ImGui::SetNextWindowSize(vp->WorkSize);
        ImGui::Begin("root",nullptr,ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoCollapse|ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoBringToFrontOnFocus|ImGuiWindowFlags_NoScrollbar);
        header();
        ImGui::BeginChild("body",ImVec2(0,0),false);
        ImGui::BeginChild("side",ImVec2(312,0),true); sidebar(d); ImGui::EndChild(); ImGui::SameLine();
        ImGui::BeginChild("content",ImVec2(0,0),true); tabs();
        ImGui::BeginChild("va",ImVec2(0,0),false);
        switch(view){ case V_DASH:view_dash(d);break; case V_TRAIN:view_train(d);break; case V_INFER:view_infer(d);break; case V_FILTERS:view_filters(d);break; case V_LOG:view_log(d);break; default:view_dash(d);break; }
        ImGui::EndChild(); ImGui::EndChild(); ImGui::EndChild();
        ImGui::End();

        ImGui::Render(); int dw,dh; glfwGetFramebufferSize(win,&dw,&dh); glViewport(0,0,dw,dh);
        ImVec4 bg=ImGui::GetStyle().Colors[ImGuiCol_WindowBg]; glClearColor(bg.x,bg.y,bg.z,1); glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData()); glfwSwapBuffers(win);
    }
    g_alive=false; g_stop=true; worker.join();
    ImGui_ImplOpenGL3_Shutdown(); ImGui_ImplGlfw_Shutdown(); ImPlot::DestroyContext(); ImGui::DestroyContext();
    glfwDestroyWindow(win); glfwTerminate(); return 0;
}
