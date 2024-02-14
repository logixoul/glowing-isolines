#include "precompiled.h"
#include "util.h"
#include "gpgpu.h"
#include "stefanfw.h"
#include "stuff.h"
#include "Array2D_imageProc.h"
#include "cfg1.h"
#include "easyfft.h"
//#include "cvstuff.h"
//#include "CrossThreadCallQueue.h"

int wsx = 600, wsy = 400;
int scale = 3;
int sx = wsx / ::scale;
int sy = wsy / ::scale;
Array2D<float> img(sx, sy);
Array2D<float> imgChange(sx, sy, 0.0f);
Array2D<float> imgChangeAcc(sx, sy, 0.0f);
bool pause = false, pause2 = false;

gl::VboMeshRef	mVboMesh;
gl::TextureRef	mTexture;
gl::GlslProgRef		mGlsl;

CameraPersp		mCamera;
//CameraUi		mCamUi;
//typedef vec2 Complexf;

Array2D<Complexf> getFdKernel(ivec2 size) {
	Array2D<float> sdKernel(size);
	forxy(sdKernel) {
		auto p2 = p; if (p2.x > sdKernel.w / 2)p2.x -= sdKernel.w; if (p2.y > sdKernel.h / 2)p2.y -= sdKernel.h;
		//sdKernel(p) = 1.0 / (.01f + (p2-ivec2(3, 3)).length()/10.0f);
		//sdKernel(p) = 1.0 / (1.f + p2.length()/10.0f);
		float dist = length(vec2(p2));
		//sdKernel(p) = powf(max(1.0f - p2.length() / 20.0f, 0.0f), 4.0);
		//sdKernel(p) = 1.0 / pow((1.f + dist*5.0f), 3.0f);
		sdKernel(p) = dist > 10 ? 0 : 1;
		//sdKernel(p) = expf(-p2.lengthSquared()*.02f);
		//if(p == ivec2::zero()) sdKernel(p) = 1.0f;
		//else sdKernel(p) = 0.0f;
	}
	auto kernelInvSum = 1.0 / (::accumulate(sdKernel.begin(), sdKernel.end(), 0.0f));
	forxy(sdKernel) { sdKernel(p) *= kernelInvSum; }
	auto fdKernel = fft(sdKernel, FFTW_MEASURE);
	return fdKernel;
}
Array2D<float> convolveLongtail(Array2D<float> in) {
	static Array2D<Complexf> fdKernel = getFdKernel(in.Size());
	auto inChanFd = fft(in, FFTW_MEASURE);
	//renderComplexImg(inChanFd);
	forxy(inChanFd) {
		auto p2 = p; if (p2.x > in.w / 2)p2.x -= in.w; if (p2.y > in.h / 2)p2.y -= in.h;
		inChanFd(p) *= fdKernel(p);
	}
	return ifft(inChanFd, FFTW_MEASURE);
}
Array2D<vec3> convolveLongtail(Array2D<vec3> in) {
	auto inChans = ::split(in);
	for (int i = 0; i < inChans.size(); i++) {
		auto& inChan = inChans[i];
		inChan = convolveLongtail(inChan);
		
	}
	return ::merge(inChans);
}

struct SApp : App {
	void setup()
	{
		setWindowSize(wsx, wsy);
		enableDenormalFlushToZero();
		disableGLReadClamp();
		reset();
		

		stefanfw::eventHandler.subscribeToEvents(*this);

		// create some geometry using a geom::Plane
		auto plane = geom::Plane().size(vec2(1, 1)).subdivisions(ivec2(sx-1, sy-1));

		vector<gl::VboMesh::Layout> bufferLayout = {
			gl::VboMesh::Layout().usage(GL_DYNAMIC_DRAW).attrib(geom::Attrib::POSITION, 3),
			gl::VboMesh::Layout().usage(GL_DYNAMIC_DRAW).attrib(geom::Attrib::NORMAL, 3),
		};

		mVboMesh = gl::VboMesh::create(plane, bufferLayout);
		mTexture = gl::Texture::create(loadImage(loadAsset("tex.png")), gl::Texture::Format().loadTopDown());

		mGlsl = gl::GlslProg::create(gl::GlslProg::Format()
			.vertex(CI_GLSL(150,
				uniform mat4	ciModelViewProjection;
				uniform mat3 ciNormalMatrix;
				in vec4			ciPosition;
				in vec3			ciNormal;
				out vec3 vNormal;
				out vec3 vVertex;
				
				void main(void) {
					//vec4 pos = ciPosition;
					//pos.y = texture2D(
					vNormal = ciNormalMatrix * ciNormal;
					gl_Position = ciModelViewProjection * ciPosition;
				}
			))
			.fragment(CI_GLSL(150,
				out vec4			oColor;
				in vec3 vNormal;

				void main(void) {
					/*vec3 L = normalize(-vVertex.xyz);
					vec3 E = normalize(-vVertex.xyz);
					vec3 R = normalize(-reflect(L, vNormal));*/

					oColor = vec4(vNormal*.5+.5, 1);
				}
			)));
	}
	void update()
	{
		stefanfw::beginFrame();
		stefanUpdate();
		stefanDraw();
		stefanfw::endFrame();
	}
	void reset()
	{
		forxy(img)
		{
			img(p) = ::randFloat();
		}
	}
	void keyDown(KeyEvent e)
	{
		if (e.getChar() == 'r')
		{
			reset();
		}
		if (e.getChar() == 'p')
		{
			pause = !pause;
		}
	}
	void stefanUpdate() {
		if (pause)
			return;
		forxy(img) {
			imgChange(p) += imgChangeAcc(p);
			img(p) += imgChange(p);
			img(p) *= .99f;
			imgChange(p) *= .99f;
		}
		//imgChange = to01(imgChange);
		
		//img = ::gauss3(img);
		//auto imgChangeB = ::gauss3(imgChange);
		//imgChange = imgChangeB;
		//forxy(imgChangeB) { imgChange(

		forxy(img) {
			//img(p) = mix(img(p), smoothstep(0.0f, 1.0f, img(p)), .3f);
		}
		static int t = 0;
		if (t++ % 100 == 0) {
			cout << " hey " << endl;
			forxy(img)
			{
				imgChangeAcc(p) = ::randFloat() * 2 - 1;
				imgChangeAcc(p) *= 10.f;
			}
		}
	}
	void stefanDraw()
	{
		gl::clear(Color(0, 0, 0));

		auto img2R = convolveLongtail(img);
		img2R = ::to01(img2R); // TODO this shouldn't be necessary
		/*
		auto tex = gtex(img2R);
		if(keys['t'])tex = shade2(tex,
			"float f = fetch1();"
			"float fw = fwidth(f);"
			"f = smoothstep(.5 - fw / 2, 0.5 + fw / 2, f);"
			"_out.r = f;"
			, ShadeOpts().scale(::scale)
		);
		
		tex = redToLuminance(tex);

		gl::draw(tex, getWindowBounds());
		*/
		// from vbo sample

		float offset = getElapsedSeconds() * 4.0f;

		// Dynmaically generate our new positions based on a sin(x) + cos(z) wave
		// We set 'orphanExisting' to false so that we can also read from the position buffer, though keep
		// in mind that this isn't the most efficient way to do cpu-side updates. Consider using VboMesh::bufferAttrib() as well.
		auto mappedPosAttrib = mVboMesh->mapAttrib3f(geom::Attrib::POSITION, false);
		forxy(img2R) {
			vec3& pos = *mappedPosAttrib;
			mappedPosAttrib->y = img2R(p) * expRange(mouseX, .1, 100);
			++mappedPosAttrib;
		}
		mappedPosAttrib.unmap();

		mCamera.setEyePoint(vec3(2, 2, 2));
		mCamera.lookAt(vec3());
		gl::setMatrices(mCamera);

		gl::ScopedGlslProg glslScope(mGlsl);
		gl::ScopedTextureBind texScope(mTexture);
		
		gl::draw(mVboMesh);
	}
	void cleanup() override {
	}
	gl::TextureRef redToLuminance(gl::TextureRef in) {
		return shade2(in, "float f = fetch1(); _out.rgb=vec3(f);", ShadeOpts().ifmt(GL_RGB8));
	}
};

class CrossThreadCallQueue* gMainThreadCallQueue;
CINDER_APP(SApp, RendererGl(),
	[&](ci::app::App::Settings *settings)
{
	//bool developer = (bool)ifstream(getAssetPath("developer"));
	//settings->setConsoleWindowEnabled(true);
})