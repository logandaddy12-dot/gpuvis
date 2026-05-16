/*
 * GPUVIS — GPU usage visualizer with particles
 *
 * Build (Arch Linux):
 *   sudo pacman -S glfw-x11 glew
 *   g++ -O2 -o gpuvis gpuvis.cpp $(pkg-config --libs --cflags glfw3 glew) -lGL -std=c++17 -lm -lpthread
 *
 * Requires nvidia-smi (sudo pacman -S nvidia-utils)
 *
 * The particle system reacts to your GPU in real time:
 *   - GPU load  → particle count, speed, chaos
 *   - VRAM use  → color shift (cool → hot)
 *   - Temp      → glow intensity
 *   - Power     → explosion force
 */

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <sstream>

// ── GPU stats (read from nvidia-smi in background thread) ────────────────────
struct GpuStats {
    float load    = 0;   // 0-100
    float vramPct = 0;   // 0-100
    float temp    = 0;   // celsius
    float power   = 0;   // watts
    float vramUsed= 0;   // MB
    float vramTotal=0;   // MB
    std::string name = "GPU";
};

static GpuStats      g_stats;
static std::mutex    g_statsMtx;
static std::atomic<bool> g_running{true};

static void statsThread() {
    while (g_running) {
        FILE* f = popen(
            "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,"
            "temperature.gpu,power.draw,memory.used,memory.total,name "
            "--format=csv,noheader,nounits 2>/dev/null", "r");
        if (f) {
            char buf[512] = {};
            if (fgets(buf, sizeof(buf), f)) {
                GpuStats s;
                char name[256] = {};
                sscanf(buf, "%f, %f, %f, %f, %f, %f, %255[^\n]",
                       &s.load, &s.vramPct, &s.temp, &s.power,
                       &s.vramUsed, &s.vramTotal, name);
                s.name = name;
                if (s.vramTotal > 0)
                    s.vramPct = s.vramUsed / s.vramTotal * 100.0f;
                std::lock_guard<std::mutex> lk(g_statsMtx);
                g_stats = s;
            }
            pclose(f);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(800));
    }
}

// ── shaders ───────────────────────────────────────────────────────────────────
const char* VS_UPDATE = R"glsl(
#version 330 core
layout(location=0) in vec2 iPos;
layout(location=1) in vec2 iVel;
layout(location=2) in float iLife;
layout(location=3) in float iSeed;

out vec2  oPos;
out vec2  oVel;
out float oLife;
out float oSeed;

uniform vec2  uRes;
uniform float uDT;
uniform float uTime;
uniform float uLoad;    // 0-1 GPU load
uniform float uVram;    // 0-1 VRAM
uniform float uTemp;    // 0-1 temp (40-95c range)
uniform float uPower;   // 0-1 power
uniform vec2  uCenter;

void main(){
    vec2 pos = iPos;
    vec2 vel = iVel;
    float life = iLife;
    float seed = iSeed;

    // ── turbulence driven by GPU load ──────────────────────────────────────
    float chaos = 0.3 + uLoad * 2.5;
    float t = uTime * (0.2 + uLoad * 0.8);
    float nx = pos.x * 0.004 + t + seed;
    float ny = pos.y * 0.004 + t * 0.73 + seed * 0.5;
    vec2 curl = vec2(-sin(ny + cos(nx)), cos(nx + sin(ny)));
    vel += curl * chaos * uDT * 60.0;

    // ── center attraction scaled by inverse load ───────────────────────────
    // low load = calm, drift to center
    // high load = chaotic, escape center
    vec2 toCenter = uCenter - pos;
    float dc = length(toCenter) + 0.001;
    float attract = (1.0 - uLoad) * 0.8 - uLoad * 0.3;
    vel += normalize(toCenter) * attract * uDT * 60.0;

    // ── VRAM drives swirl ──────────────────────────────────────────────────
    float swirl = uVram * 2.0;
    vec2 perp = vec2(-toCenter.y, toCenter.x) / (dc + 1.0);
    vel += perp * swirl * uDT * 60.0;

    // ── temp drives outward pulse ──────────────────────────────────────────
    float pulse = sin(uTime * (2.0 + uTemp * 8.0)) * uTemp * 1.5;
    vel += normalize(pos - uCenter + vec2(0.001)) * pulse * uDT * 60.0;

    // ── damping ───────────────────────────────────────────────────────────
    float damp = 0.97 - uLoad * 0.05;
    vel *= pow(damp, uDT * 60.0);

    // ── speed cap ─────────────────────────────────────────────────────────
    float maxSpd = 200.0 + uLoad * 600.0 + uPower * 300.0;
    float spd = length(vel);
    if(spd > maxSpd) vel = vel / spd * maxSpd;

    pos += vel * uDT;

    // wrap
    if(pos.x < 0.0) pos.x += uRes.x;
    if(pos.x > uRes.x) pos.x -= uRes.x;
    if(pos.y < 0.0) pos.y += uRes.y;
    if(pos.y > uRes.y) pos.y -= uRes.y;

    life = min(life + uDT * 0.3, 1.0);

    oPos  = pos;
    oVel  = vel;
    oLife = life;
    oSeed = seed;
}
)glsl";

const char* FS_EMPTY = R"glsl(#version 330 core
void main(){}
)glsl";

const char* VS_DRAW = R"glsl(
#version 330 core
layout(location=0) in vec2 iPos;
layout(location=1) in vec2 iVel;
layout(location=2) in float iLife;
layout(location=3) in float iSeed;

out vec4 vColor;

uniform vec2  uRes;
uniform float uTime;
uniform float uLoad;
uniform float uVram;
uniform float uTemp;
uniform float uPower;

vec3 hsv(float h,float s,float v){
    vec3 p=abs(fract(vec3(h)+vec3(1,2,3)/3.0)*6.0-3.0);
    return v*mix(vec3(1),clamp(p-1.0,0.0,1.0),s);
}

void main(){
    vec2 ndc = (iPos/uRes)*2.0-1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc,0,1);

    float spd = length(iVel);
    float normSpd = clamp(spd/500.0, 0.0, 1.0);

    // idle=cool blue, medium=purple, high=RED INSANE
    float hue = mix(0.62, 0.0, smoothstep(0.3, 1.0, uLoad))  // blue -> red
              - uVram*0.08
              + sin(iSeed*6.28 + uTime*uLoad*4.0)*0.04*uLoad; // jitter at high load
    float sat = 0.6 + normSpd*0.3 + uLoad*0.4;
    float val = 0.3 + normSpd*0.6 + uLoad*0.5;

    // white hot core when really cooking
    float whiteness = smoothstep(0.7,1.0,uLoad) * normSpd * 2.0
                    + uTemp * normSpd * 1.0;
    vec3 col = mix(hsv(hue,sat,val), vec3(1.0,0.9,0.8), clamp(whiteness,0,1));

    // crazy flickering sparks at high load
    float sparkle = sin(iSeed*73.1 + uTime*(8.0+uLoad*30.0))*0.5+0.5;
    float sparkStr = uLoad*uLoad*0.8 + uPower*0.3;
    col += sparkle * sparkStr * vec3(1.0,0.3,0.1);

    // at very high load particles flash randomly
    float flash = step(0.97, sin(iSeed*17.3 + uTime*50.0*uLoad));
    col = mix(col, vec3(1.0,0.1,0.0), flash*smoothstep(0.7,1.0,uLoad));

    float alpha = 0.5 + normSpd*0.5 + uLoad*0.3;
    vColor = vec4(col, min(alpha,1.0));

    // point size explodes with load
    float crazySize = smoothstep(0.6,1.0,uLoad)*3.0;
    gl_PointSize = 1.5 + normSpd*2.5 + uLoad*2.0 + crazySize
                 + sin(iSeed*5.1+uTime*20.0*uLoad)*crazySize*0.5;
}
)glsl";

const char* FS_DRAW = R"glsl(
#version 330 core
in vec4 vColor;
out vec4 oColor;
void main(){
    vec2 c = gl_PointCoord*2.0-1.0;
    float r = dot(c,c);
    if(r>1.0) discard;
    oColor = vec4(vColor.rgb, vColor.a*(1.0-r*0.8));
}
)glsl";

const char* VS_QUAD = R"glsl(
#version 330 core
layout(location=0) in vec2 aPos;
out vec2 vUV;
void main(){ gl_Position=vec4(aPos,0,1); vUV=aPos*0.5+0.5; }
)glsl";

const char* FS_TRAIL = R"glsl(
#version 330 core
in vec2 vUV; out vec4 oC;
uniform sampler2D uTex;
uniform float uDecay;
uniform float uLoad;
void main(){
    vec2 res = vec2(textureSize(uTex,0));
    // slight motion blur in direction of load
    vec4 c = texture(uTex, vUV);
    vec4 blur=vec4(0); float w=0;
    for(int x=-1;x<=1;x++) for(int y=-1;y<=1;y++){
        float wt=1.0/(1.0+length(vec2(x,y)));
        blur+=texture(uTex,vUV+vec2(x,y)/res)*wt; w+=wt;
    }
    blur/=w;
    oC = max(c, blur*0.3) * uDecay;
}
)glsl";

const char* FS_COMPOSITE = R"glsl(
#version 330 core
in vec2 vUV; out vec4 oC;
uniform sampler2D uScene;
uniform sampler2D uTrail;
uniform float uTime;
uniform float uLoad;
uniform float uTemp;
uniform float uVram;
uniform float uPower;
uniform vec2  uRes;

void main(){
    vec4 scene = texture(uScene, vUV);
    vec4 trail = texture(uTrail, vUV);

    vec3 col = scene.rgb*1.2 + trail.rgb*0.9;

    // bloom glow: sample neighbors
    vec2 res = uRes;
    vec3 bloom=vec3(0);
    float bw=0;
    for(int x=-3;x<=3;x++) for(int y=-3;y<=3;y++){
        if(x==0&&y==0) continue;
        float d=length(vec2(x,y));
        float wt=1.0/(d*d+0.5);
        bloom+=texture(uScene,vUV+vec2(x,y)/res).rgb*wt;
        bw+=wt;
    }
    bloom/=bw;
    float bloomStr=0.4+uLoad*0.8+uTemp*0.4;
    col += bloom*bloomStr;

    // tone map
    col = col/(col+0.6);
    col = pow(col, vec3(0.85));

    // vignette
    vec2 uv=vUV*2.0-1.0;
    float vig=1.0-dot(uv,uv)*0.4;
    col*=vig;

    // scanline flicker — gets intense at high load
    float scanIntensity = uTemp*0.04 + uLoad*uLoad*0.08;
    float scan=sin(vUV.y*uRes.y*3.14159)*scanIntensity;
    col-=scan;

    // chromatic aberration — goes wild at high load
    float abr = uPower*0.003 + uLoad*uLoad*0.012;
    vec2 aberUV_r = vUV + vec2(abr,  sin(uTime*3.0)*abr*0.3);
    vec2 aberUV_b = vUV - vec2(abr, -sin(uTime*3.0)*abr*0.3);
    col.r = texture(uScene,aberUV_r).r*1.3 + texture(uTrail,aberUV_r).r*0.9;
    col.b = texture(uScene,aberUV_b).b*1.3 + texture(uTrail,aberUV_b).b*0.9;
    col = col/(col+0.6);
    col = pow(col,vec3(0.85));
    col *= vig;

    // RED TINT when going hard
    float redLevel = smoothstep(0.5, 1.0, uLoad);
    col = mix(col, col*vec3(2.0,0.3,0.2), redLevel*0.45);

    // screen-wide red pulse at very high load
    float pulse = sin(uTime*(3.0+uLoad*15.0))*0.5+0.5;
    float danger = smoothstep(0.75, 1.0, uLoad);
    col += vec3(0.4,0.0,0.0)*pulse*danger;

    // glitch: horizontal slice displacement
    float glitchStr = smoothstep(0.6,1.0,uLoad);
    float sliceY = fract(uTime*7.0*uLoad);
    float sliceH = 0.03*uLoad;
    if(abs(vUV.y - sliceY) < sliceH){
        float shift = (sin(uTime*31.0)*0.5+0.5)*0.04*glitchStr;
        col = texture(uScene, vec2(vUV.x+shift, vUV.y)).rgb
            + texture(uTrail, vec2(vUV.x+shift, vUV.y)).rgb*0.8;
        col = col/(col+0.6);
        col *= vec3(1.5,0.4,0.4)*glitchStr + vec3(1)*(1.0-glitchStr);
    }

    // random static blocks at 90%+
    float block = step(0.985, fract(sin(floor(vUV.y*40.0)*127.1+uTime*60.0)*43758.5));
    col = mix(col, vec3(1.0,0.0,0.0)*pulse, block*smoothstep(0.85,1.0,uLoad));

    oC=vec4(col,1.0);
}
)glsl";

// HUD overlay (stats text)
const char* FS_HUD = R"glsl(
#version 330 core
in vec2 vUV; out vec4 oC;
uniform float uLoad;
uniform float uVram;
uniform float uTemp;
uniform float uPower;
uniform float uTime;
// We draw bars procedurally
void main(){
    vec2 uv=vUV;
    // draw 4 bars in bottom-left corner
    // each bar: x 0.01-0.25, y stacked
    float barX0=0.01, barW=0.24;
    float barH=0.018, barGap=0.026;

    float vals[4]; vals[0]=uLoad; vals[1]=uVram; vals[2]=uTemp; vals[3]=uPower;
    vec3 cols[4];
    cols[0]=vec3(0.2,0.8,1.0);   // load: blue
    cols[1]=vec3(1.0,0.5,0.1);   // vram: orange
    cols[2]=vec3(1.0,0.2,0.2);   // temp: red
    cols[3]=vec3(0.5,1.0,0.3);   // power: green

    for(int i=0;i<4;i++){
        float y0=0.04+i*barGap;
        float y1=y0+barH;
        if(uv.x>barX0 && uv.x<barX0+barW && uv.y>y0 && uv.y<y1){
            float fill=barX0+vals[i]*barW;
            float edge=abs(uv.x-(barX0+vals[i]*barW));
            if(uv.x<fill){
                // pulse glow on leading edge
                float glow=exp(-edge*200.0)*0.6;
                vec3 c=cols[i]+glow;
                float flicker=sin(uTime*20.0+i)*0.05;
                oC=vec4(c+flicker,0.85); return;
            } else {
                oC=vec4(cols[i]*0.08,0.6); return;
            }
        }
        // bar border
        float bord=0.001;
        if(uv.x>barX0-bord&&uv.x<barX0+barW+bord&&uv.y>y0-bord&&uv.y<y1+bord){
            oC=vec4(0.3,0.3,0.3,0.5); return;
        }
    }
    oC=vec4(0,0,0,0);
}
)glsl";

// ── utils ─────────────────────────────────────────────────────────────────────
static GLuint compSh(GLenum t, const char* s){
    GLuint sh=glCreateShader(t);
    glShaderSource(sh,1,&s,nullptr);
    glCompileShader(sh);
    GLint ok; glGetShaderiv(sh,GL_COMPILE_STATUS,&ok);
    if(!ok){char l[1024];glGetShaderInfoLog(sh,1024,nullptr,l);std::cerr<<l<<"\n";}
    return sh;
}
static GLuint linkProg(const char* v,const char* f,
    const std::vector<const char*>& varyings={}){
    GLuint p=glCreateProgram();
    glAttachShader(p,compSh(GL_VERTEX_SHADER,v));
    glAttachShader(p,compSh(GL_FRAGMENT_SHADER,f));
    if(!varyings.empty())
        glTransformFeedbackVaryings(p,(GLsizei)varyings.size(),varyings.data(),GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    GLint ok; glGetProgramiv(p,GL_LINK_STATUS,&ok);
    if(!ok){char l[1024];glGetProgramInfoLog(p,1024,nullptr,l);std::cerr<<l<<"\n";}
    return p;
}

static GLuint quadVAO,quadVBO;
static void initQuad(){
    float v[]={-1,-1, 1,-1, 1,1, -1,-1, 1,1, -1,1};
    glGenVertexArrays(1,&quadVAO); glGenBuffers(1,&quadVBO);
    glBindVertexArray(quadVAO); glBindBuffer(GL_ARRAY_BUFFER,quadVBO);
    glBufferData(GL_ARRAY_BUFFER,sizeof(v),v,GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,0,nullptr);
    glBindVertexArray(0);
}
static void drawQuad(){ glBindVertexArray(quadVAO); glDrawArrays(GL_TRIANGLES,0,6); }

struct FBO{
    GLuint fbo=0,tex=0,w=0,h=0;
    void init(int W,int H){
        w=W;h=H;
        if(fbo) glDeleteFramebuffers(1,&fbo);
        if(tex) glDeleteTextures(1,&tex);
        glGenFramebuffers(1,&fbo); glBindFramebuffer(GL_FRAMEBUFFER,fbo);
        glGenTextures(1,&tex); glBindTexture(GL_TEXTURE_2D,tex);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,W,H,0,GL_RGBA,GL_FLOAT,nullptr);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,tex,0);
        glBindFramebuffer(GL_FRAMEBUFFER,0);
    }
    void bind(int W=-1,int H=-1){
        glBindFramebuffer(GL_FRAMEBUFFER,fbo);
        if(W>0) glViewport(0,0,W,H);
    }
    static void unbind(int W=0,int H=0){
        glBindFramebuffer(GL_FRAMEBUFFER,0);
        if(W>0) glViewport(0,0,W,H);
    }
};

// ── particle struct ───────────────────────────────────────────────────────────
struct Particle { float x,y,vx,vy,life,seed; };
static const int NUM_P = 60000;

// ── input ─────────────────────────────────────────────────────────────────────
static int gW=1280,gH=720;
static bool gResize=false;
void cbKey(GLFWwindow*w,int k,int,int a,int){if(k==GLFW_KEY_ESCAPE&&a==GLFW_PRESS)glfwSetWindowShouldClose(w,1);}
void cbResize(GLFWwindow*,int w,int h){gW=w;gH=h;gResize=true;}

// ── main ──────────────────────────────────────────────────────────────────────
int main(){
    srand((unsigned)time(nullptr));

    // start stats thread
    std::thread(statsThread).detach();

    if(!glfwInit()){std::cerr<<"GLFW\n";return 1;}
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES,1);

    auto*win=glfwCreateWindow(gW,gH,"GPU VISUALIZER",nullptr,nullptr);
    if(!win){glfwTerminate();return 1;}
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);
    glfwSetKeyCallback(win,cbKey);
    glfwSetFramebufferSizeCallback(win,cbResize);

    glewExperimental=GL_TRUE; glewInit();
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);

    // programs
    GLuint pUpdate =linkProg(VS_UPDATE,FS_EMPTY,{"oPos","oVel","oLife","oSeed"});
    GLuint pDraw   =linkProg(VS_DRAW,  FS_DRAW);
    GLuint pTrail  =linkProg(VS_QUAD,  FS_TRAIL);
    GLuint pComp   =linkProg(VS_QUAD,  FS_COMPOSITE);
    GLuint pHUD    =linkProg(VS_QUAD,  FS_HUD);
    initQuad();

    FBO fboScene,fboTrail,fboTemp;
    fboScene.init(gW,gH);
    fboTrail.init(gW,gH);
    fboTemp.init(gW,gH);

    // clear trail
    fboTrail.bind(gW,gH);
    glClearColor(0,0,0,1);glClear(GL_COLOR_BUFFER_BIT);
    FBO::unbind(gW,gH);

    // init particles
    std::vector<Particle> particles(NUM_P);
    float cx=(float)gW*0.5f, cy=(float)gH*0.5f;
    for(auto&p:particles){
        float a=(float)rand()/RAND_MAX*6.2831f;
        float r=(float)rand()/RAND_MAX*(float)std::min(gW,gH)*0.4f;
        p.x=cx+cosf(a)*r;
        p.y=cy+sinf(a)*r;
        p.vx=((float)rand()/RAND_MAX-0.5f)*80;
        p.vy=((float)rand()/RAND_MAX-0.5f)*80;
        p.life=0;
        p.seed=(float)rand()/RAND_MAX*100.0f;
    }

    // GPU buffers (ping-pong)
    GLuint vbo[2],vao[2],tbo[2];
    glGenBuffers(2,vbo);
    glGenVertexArrays(2,vao);
    glGenTransformFeedbacks(2,tbo);
    for(int i=0;i<2;i++){
        glBindBuffer(GL_ARRAY_BUFFER,vbo[i]);
        glBufferData(GL_ARRAY_BUFFER,NUM_P*sizeof(Particle),particles.data(),GL_DYNAMIC_COPY);
        glBindVertexArray(vao[i]);
        glBindBuffer(GL_ARRAY_BUFFER,vbo[i]);
        glEnableVertexAttribArray(0); glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,sizeof(Particle),(void*)0);
        glEnableVertexAttribArray(1); glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,sizeof(Particle),(void*)(2*4));
        glEnableVertexAttribArray(2); glVertexAttribPointer(2,1,GL_FLOAT,GL_FALSE,sizeof(Particle),(void*)(4*4));
        glEnableVertexAttribArray(3); glVertexAttribPointer(3,1,GL_FLOAT,GL_FALSE,sizeof(Particle),(void*)(5*4));
        glBindTransformFeedback(GL_TRANSFORM_FEEDBACK,tbo[i]);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER,0,vbo[i]);
    }
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK,0);

    int cur=0;
    double lastT=glfwGetTime();

    // smoothed stats for nice transitions
    float sLoad=0,sVram=0,sTemp=0,sPower=0;

    while(!glfwWindowShouldClose(win)){
        glfwPollEvents();

        if(gResize){
            gResize=false;
            fboScene.init(gW,gH);
            fboTrail.init(gW,gH);
            fboTemp.init(gW,gH);
            fboTrail.bind(gW,gH);
            glClearColor(0,0,0,1);glClear(GL_COLOR_BUFFER_BIT);
            FBO::unbind(gW,gH);
        }

        double now=glfwGetTime();
        float dt=std::min((float)(now-lastT),0.033f);
        lastT=now;
        float t=(float)now;

        // read stats
        float rawLoad,rawVram,rawTemp,rawPower;
        {std::lock_guard<std::mutex>lk(g_statsMtx);
         rawLoad =g_stats.load/100.0f;
         rawVram =g_stats.vramPct/100.0f;
         rawTemp =std::clamp((g_stats.temp-35.0f)/65.0f,0.0f,1.0f);
         rawPower=std::clamp(g_stats.power/350.0f,0.0f,1.0f);}

        // smooth
        float smooth=std::min(dt*2.5f,1.0f);
        sLoad =sLoad +(rawLoad -sLoad )*smooth;
        sVram =sVram +(rawVram -sVram )*smooth;
        sTemp =sTemp +(rawTemp -sTemp )*smooth;
        sPower=sPower+(rawPower-sPower)*smooth;

        glViewport(0,0,gW,gH);
        int next=1-cur;

        // ── update ────────────────────────────────────────────────────────────
        glEnable(GL_RASTERIZER_DISCARD);
        glUseProgram(pUpdate);
        glUniform2f(glGetUniformLocation(pUpdate,"uRes"),(float)gW,(float)gH);
        glUniform1f(glGetUniformLocation(pUpdate,"uDT"),dt);
        glUniform1f(glGetUniformLocation(pUpdate,"uTime"),t);
        glUniform1f(glGetUniformLocation(pUpdate,"uLoad"),sLoad);
        glUniform1f(glGetUniformLocation(pUpdate,"uVram"),sVram);
        glUniform1f(glGetUniformLocation(pUpdate,"uTemp"),sTemp);
        glUniform1f(glGetUniformLocation(pUpdate,"uPower"),sPower);
        glUniform2f(glGetUniformLocation(pUpdate,"uCenter"),(float)gW*0.5f,(float)gH*0.5f);

        glBindVertexArray(vao[cur]);
        glBindTransformFeedback(GL_TRANSFORM_FEEDBACK,tbo[next]);
        glBeginTransformFeedback(GL_POINTS);
        glDrawArrays(GL_POINTS,0,NUM_P);
        glEndTransformFeedback();
        glBindTransformFeedback(GL_TRANSFORM_FEEDBACK,0);
        glDisable(GL_RASTERIZER_DISCARD);

        // ── draw particles → scene FBO ────────────────────────────────────────
        fboScene.bind(gW,gH);
        glClearColor(0,0,0,0);glClear(GL_COLOR_BUFFER_BIT);
        glBlendFunc(GL_SRC_ALPHA,GL_ONE);
        glUseProgram(pDraw);
        glUniform2f(glGetUniformLocation(pDraw,"uRes"),(float)gW,(float)gH);
        glUniform1f(glGetUniformLocation(pDraw,"uTime"),t);
        glUniform1f(glGetUniformLocation(pDraw,"uLoad"),sLoad);
        glUniform1f(glGetUniformLocation(pDraw,"uVram"),sVram);
        glUniform1f(glGetUniformLocation(pDraw,"uTemp"),sTemp);
        glUniform1f(glGetUniformLocation(pDraw,"uPower"),sPower);
        glBindVertexArray(vao[next]);
        glDrawArrays(GL_POINTS,0,NUM_P);
        FBO::unbind(gW,gH);

        // ── decay trail ───────────────────────────────────────────────────────
        float decay=0.82f+sLoad*0.10f; // high load = longer trails
        fboTemp.bind(gW,gH);
        glClearColor(0,0,0,0);glClear(GL_COLOR_BUFFER_BIT);
        glBlendFunc(GL_ONE,GL_ZERO);
        glUseProgram(pTrail);
        glActiveTexture(GL_TEXTURE0);glBindTexture(GL_TEXTURE_2D,fboTrail.tex);
        glUniform1i(glGetUniformLocation(pTrail,"uTex"),0);
        glUniform1f(glGetUniformLocation(pTrail,"uDecay"),decay);
        glUniform1f(glGetUniformLocation(pTrail,"uLoad"),sLoad);
        drawQuad();
        // add scene
        glBlendFunc(GL_ONE,GL_ONE);
        glBindTexture(GL_TEXTURE_2D,fboScene.tex);
        glUniform1f(glGetUniformLocation(pTrail,"uDecay"),1.0f);
        drawQuad();
        FBO::unbind(gW,gH);

        // copy temp→trail
        fboTrail.bind(gW,gH);
        glClearColor(0,0,0,0);glClear(GL_COLOR_BUFFER_BIT);
        glBlendFunc(GL_ONE,GL_ZERO);
        glUseProgram(pTrail);
        glBindTexture(GL_TEXTURE_2D,fboTemp.tex);
        glUniform1f(glGetUniformLocation(pTrail,"uDecay"),1.0f);
        drawQuad();
        FBO::unbind(gW,gH);

        // ── composite ─────────────────────────────────────────────────────────
        glBindFramebuffer(GL_FRAMEBUFFER,0);
        glViewport(0,0,gW,gH);
        glClearColor(0,0,0,1);glClear(GL_COLOR_BUFFER_BIT);
        glBlendFunc(GL_ONE,GL_ZERO);
        glUseProgram(pComp);
        glActiveTexture(GL_TEXTURE0);glBindTexture(GL_TEXTURE_2D,fboScene.tex);
        glUniform1i(glGetUniformLocation(pComp,"uScene"),0);
        glActiveTexture(GL_TEXTURE1);glBindTexture(GL_TEXTURE_2D,fboTrail.tex);
        glUniform1i(glGetUniformLocation(pComp,"uTrail"),1);
        glUniform1f(glGetUniformLocation(pComp,"uTime"),t);
        glUniform1f(glGetUniformLocation(pComp,"uLoad"),sLoad);
        glUniform1f(glGetUniformLocation(pComp,"uTemp"),sTemp);
        glUniform1f(glGetUniformLocation(pComp,"uVram"),sVram);
        glUniform1f(glGetUniformLocation(pComp,"uPower"),sPower);
        glUniform2f(glGetUniformLocation(pComp,"uRes"),(float)gW,(float)gH);
        drawQuad();

        // ── HUD bars ──────────────────────────────────────────────────────────
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
        glUseProgram(pHUD);
        glUniform1f(glGetUniformLocation(pHUD,"uLoad"),sLoad);
        glUniform1f(glGetUniformLocation(pHUD,"uVram"),sVram);
        glUniform1f(glGetUniformLocation(pHUD,"uTemp"),sTemp);
        glUniform1f(glGetUniformLocation(pHUD,"uPower"),sPower);
        glUniform1f(glGetUniformLocation(pHUD,"uTime"),t);
        drawQuad();

        cur=next;
        glfwSwapBuffers(win);
    }

    g_running=false;
    glfwTerminate();
    return 0;
}
