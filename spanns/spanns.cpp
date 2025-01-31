#include <algorithm>
#include <limits>
#include <vector>
#include "VapourSynth4.h"
#include "VSHelper4.h"
#include <boost/sort/pdqsort/pdqsort.hpp>
#include <Eigen/SVD>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <gsl/gsl_multimin.h>

#include "wavelib.h"


const double _XMAX = std::numeric_limits<double>::max();


typedef struct {
    VSNode *node;
    VSNode *ref;
    float sigma;
    float tol;
    float cutoff;
    bool has_ref;
} SpannsData;

/* TODO: DWT Adaptive Denoising
static void wavelet_transform(const cv::Mat& input, const char* wavelet_type, cv::Mat& details) {
    wave_object obj;
    wt2_object wt;
    
    obj = wave_init(wavelet_type);

    int rows = input.rows;
    int cols = input.cols;
    
    double* inp = (double*)malloc(sizeof(double) * rows * cols);
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            inp[i*cols + j] = input.at<float>(i,j);
        }
    }
    
    wt = wt2_init(obj, "dwt", rows, cols, 1);
    
    double* out = dwt2(wt, inp);
    
    int J = wt->J;
    int rJ = rows >> J;
    int cJ = cols >> J;
    for(int i = 0; i < rJ * cJ; i++) {
        out[i] = 0.0;
    }
    
    double* final = (double*)malloc(sizeof(double) * rows * cols);
    
    idwt2(wt, out, final);
    
    details = cv::Mat(rows, cols, CV_32F);
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            details.at<float>(i,j) = (float)final[i*cols + j];
        }
    }
    
    free(inp);
    free(out);
    free(final);
    wave_free(obj);
    wt2_free(wt);
}


static void generate_mask(const float* src, int width, int height, float gamma, float* mask) {
    cv::Mat src_mat(height, width, CV_32F);
    memcpy(src_mat.data, src, width * height * sizeof(float));
    
    cv::Mat B1, B2;
    wavelet_transform(src_mat, "bior1.1", B1);
    wavelet_transform(src_mat, "coif1", B2);
    
    cv::Mat abs_B1, abs_B2;
    cv::absdiff(B1, cv::Scalar(0), abs_B1);
    cv::absdiff(B2, cv::Scalar(0), abs_B2);
    
    cv::Mat B;
    cv::max(abs_B1, abs_B2, B);
    
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
    cv::dilate(B, B, kernel);
    
    std::vector<float> sorted_data;
    sorted_data.reserve(width * height);
    for(int i = 0; i < height; ++i) {
        const float* row = B.ptr<float>(i);
        sorted_data.insert(sorted_data.end(), row, row + width);
    }
    boost::sort::pdqsort(sorted_data.begin(), sorted_data.end());
    
    float thresh = sorted_data[static_cast<int>(gamma * (width * height - 1))];
    float min_val = sorted_data[0];
    
    if (thresh > min_val) {
        float scale = 1.0f / (thresh - min_val);
        float offset = -min_val * scale;
        B.convertTo(B, CV_32F, scale, offset);
        cv::min(B, cv::Scalar(1.0f), B); 
        cv::max(B, cv::Scalar(0.0f), B);
    } else {
        B = cv::Mat::zeros(B.size(), CV_32F);
    }
    
    memcpy(mask, B.ptr<float>(), width * height * sizeof(float));
}
*/

class MPDistribution {
private:
    double beta_;
    double sigma_;
    double ratio_;
    
    std::pair<double, double> get_lambdas() const {
        double lambda_minus = beta_ * sigma_ * sigma_ * std::pow(1 - std::sqrt(ratio_), 2);
        double lambda_plus = beta_ * sigma_ * sigma_ * std::pow(1 + std::sqrt(ratio_), 2);
        return {lambda_minus, lambda_plus};
    }
    
    double cdf_aux_r(double x, double lambda_minus, double lambda_plus) const {
        return std::sqrt((lambda_plus - x) / (x - lambda_minus));
    }
    
    double cdf_aux_f(double x) const {
        auto [lambda_minus, lambda_plus] = get_lambdas();
        double var = beta_ * sigma_ * sigma_;
        
        if (x <= lambda_minus) return 0.0;
        if (x >= lambda_plus) return 1.0;
        
        double r = cdf_aux_r(x, lambda_minus, lambda_plus);
        
        double first_arctan;
        if (x == lambda_minus) {
            first_arctan = M_PI / 2;
        } else {
            first_arctan = std::atan((r * r - 1) / (2 * r));
        }
        
        double second_arctan;
        if (x == lambda_minus) {
            second_arctan = M_PI / 2;
        } else {
            second_arctan = std::atan((lambda_minus * r * r - lambda_plus) / (2 * var * (1 - ratio_) * r));
        }
        
        double sqrt_term = std::sqrt((lambda_plus - x) * (x - lambda_minus));
        
        return 1.0 / (2 * M_PI * ratio_) * (
            M_PI * ratio_ + 
            (1.0 / var) * sqrt_term - 
            (1 + ratio_) * first_arctan + 
            (1 - ratio_) * second_arctan
        );
    }

public:
    MPDistribution(double beta, double sigma, double ratio) 
        : beta_(beta), sigma_(sigma), ratio_(ratio) {}
    
    double cdf(double x) const {
        auto [lambda_minus, lambda_plus] = get_lambdas();
        
        if (x >= lambda_plus) return 1.0;
        if (x <= lambda_minus) return 0.0;
        
        double result = cdf_aux_f(x);
        
        if (ratio_ <= 1.0) {
            return result;
        }
        
        if (x >= lambda_minus && x <= lambda_plus) {
            result += (ratio_ - 1) / (2 * ratio_);
        }
        
        return std::clamp(result, 0.0, 1.0);
    }
};


struct MpFitData {
    const std::vector<double> *x; 
};


static double mp_pdf(double val, double beta, double sigma, double ratio) {
    double var = beta * sigma * sigma;
    double lambda_minus = var * std::pow(1.0 - std::sqrt(ratio), 2.0);
    double lambda_plus  = var * std::pow(1.0 + std::sqrt(ratio), 2.0);

    if (val < lambda_minus || val > lambda_plus || val <= 0.0)
        return 1e-30;

    double numerator = std::sqrt((lambda_plus - val)*(val - lambda_minus));
    double denom = 2.0 * M_PI * ratio * var * val;

    if (denom < 1e-30) denom=1e-30;

    return numerator / denom;
}


static double nll_function(const gsl_vector *v, void *params) {
    double beta = gsl_vector_get(v, 0);
    double sigma = gsl_vector_get(v, 1);
    double ratio = gsl_vector_get(v, 2);
    
    if (beta <= 1e-2 || sigma <= 1e-3 || ratio <= 1e-6 || ratio >= 1.0)
        return GSL_POSINF;

    MpFitData *d = (MpFitData *)(params);
    const std::vector<double> &x = *(d->x);

    double lambda_minus = beta * sigma * sigma * std::pow(1.0 - std::sqrt(ratio), 2);
    double lambda_plus = beta * sigma * sigma * std::pow(1.0 + std::sqrt(ratio), 2);
    
    double nll = 0.0;
    int invalid_points = 0;
    
    for (auto val : x) {
        if (val < lambda_minus) {
            invalid_points++;
        } else if (val < lambda_plus) {
            double p = mp_pdf(val, beta, sigma, ratio);
            nll -= std::log(p);
        } else {
            invalid_points++;
        }
    }

    float penalty = - std::log(_XMAX) * invalid_points * 3;  // 3 times max loss
    
    if (invalid_points > x.size() * 0.5)
        return GSL_POSINF;

    return nll + penalty;
}


std::tuple<double, double, double> fit_mp_distribution_gsl(const std::vector<double> &sorted_sv, float cutoff) {

    double ub = 0.;
    if (cutoff < 1.) {
        ub = sorted_sv[size_t(sorted_sv.size() * cutoff)];
    } else {
        ub = sorted_sv[sorted_sv.size() - size_t(cutoff)];
    }

    std::vector<double> filtered_sv;
    for (auto s : sorted_sv)
        if (s < ub)
            filtered_sv.push_back(s);

    MpFitData data;
    data.x = &filtered_sv;

    gsl_multimin_function f;
    f.f = &nll_function;
    f.n = 3;
    f.params = &data;

    gsl_vector *x_start = gsl_vector_alloc(3);

    size_t n = filtered_sv.size();
    double lm = filtered_sv[size_t(n * 0.05)];
    double lp = filtered_sv[size_t(n * 0.95)];
    double a = std::sqrt(lm);
    double b = std::sqrt(lp);
    double sigma_init = std::sqrt((a + b) * (a + b) / 4.0);
    double ratio_init = std::sqrt((b - a) / (a + b));

    if (sigma_init < 1e-5) sigma_init = 1e-5;
    if (ratio_init < 1e-5) ratio_init = 1e-5;
    if (ratio_init > 0.9999) ratio_init = 0.9999;

    double beta_init = 1.;

    gsl_vector_set(x_start, 0, beta_init);
    gsl_vector_set(x_start, 1, sigma_init);
    gsl_vector_set(x_start, 2, ratio_init);

    gsl_multimin_fminimizer *fminimizer = nullptr;
    const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;
    
    fminimizer = gsl_multimin_fminimizer_alloc(T, f.n);
    
    gsl_vector *step_size = gsl_vector_alloc(3);
    gsl_vector_set(step_size, 0, 0.1);
    gsl_vector_set(step_size, 1, sigma_init * 0.1);
    gsl_vector_set(step_size, 2, 0.1);

    int status = gsl_multimin_fminimizer_set(fminimizer, &f, x_start, step_size);

    const size_t MAX_ITER = 114514;
    size_t iter = 0;
    do {
        iter++;
        status = gsl_multimin_fminimizer_iterate(fminimizer);
        if (status) break;

        double cur_size = gsl_multimin_fminimizer_size(fminimizer);
        status = gsl_multimin_test_size(cur_size, 1e-7);

    } while (status == GSL_CONTINUE && iter < MAX_ITER);

    double beta_opt = gsl_vector_get(fminimizer->x, 0);
    double sigma_opt = gsl_vector_get(fminimizer->x, 1);
    double ratio_opt = gsl_vector_get(fminimizer->x, 2);

    gsl_multimin_fminimizer_free(fminimizer);
    gsl_vector_free(x_start);
    gsl_vector_free(step_size);

    return {beta_opt, sigma_opt, ratio_opt};
}


static std::tuple<double, double, double> fit_mp_distribution(const Eigen::VectorXf& singular_values, float cutoff) {
    std::vector<double> sv(singular_values.data(), singular_values.data() + singular_values.size());
    boost::sort::pdqsort(sv.begin(), sv.end());

    return fit_mp_distribution_gsl(sv, cutoff);
}


static void box_blur(const float* src, float* dst, int width, int height, float sigma) {
    cv::Mat src_mat(height, width, CV_32F);
    cv::Mat temp_mat;
    cv::Mat dst_mat;

    int kernal_size = static_cast<int>(std::round(sigma)) * 2 + 1; 
    
    memcpy(src_mat.data, src, width * height * sizeof(float));
    
    cv::boxFilter(src_mat, temp_mat, -1, cv::Size(kernal_size, kernal_size), 
                cv::Point(-1,-1), true, cv::BORDER_DEFAULT);
    
    cv::boxFilter(temp_mat, dst_mat, -1, cv::Size(kernal_size, kernal_size), 
                cv::Point(-1,-1), true, cv::BORDER_DEFAULT);
    
    cv::boxFilter(dst_mat, temp_mat, -1, cv::Size(kernal_size, kernal_size), 
                cv::Point(-1,-1), true, cv::BORDER_DEFAULT);
    
    memcpy(dst, temp_mat.ptr<float>(), width * height * sizeof(float));
}


static void process_plane_spanns(const float* src, ptrdiff_t src_stride,
                                const float* ref, ptrdiff_t ref_stride,
                                float* dst, ptrdiff_t dst_stride,
                                int width, int height,
                                float sigma, float tol, float cutoff) {
    if (!src || !dst) return;
    
    std::vector<float> src_buf(width * height);
    std::vector<float> ref_buf(width * height);
    std::vector<float> dst_buf(width * height);
    std::vector<float> T(width * height);
    std::vector<float> mask(width * height);
    
    for (int y = 0; y < height; y++) {
        memcpy(src_buf.data() + y * width, 
               reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(src) + y * src_stride),
               width * sizeof(float));
    }
    
    const float* ref_ptr = ref;
    
    if (ref_ptr) {
        for (int y = 0; y < height; y++) {
            memcpy(ref_buf.data() + y * width,
                   reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(ref) + y * ref_stride),
                   width * sizeof(float));
        }
        ref_ptr = ref_buf.data();
    } else {
        box_blur(src_buf.data(), ref_buf.data(), width, height, sigma);
        ref_ptr = ref_buf.data();
    }
    
    memcpy(T.data(), src_buf.data(), width * height * sizeof(float));

    Eigen::MatrixXf T_mat = Eigen::Map<const Eigen::MatrixXf>(T.data(), height, width);
    Eigen::MatrixXf ref_mat = Eigen::Map<const Eigen::MatrixXf>(ref_ptr, height, width);

    Eigen::MatrixXf noise = ref_mat - T_mat;

    if ((noise.maxCoeff() - noise.minCoeff()) < 1e-6) {
        for (int y = 0; y < height; y++) {
            memcpy(reinterpret_cast<uint8_t*>(dst) + y * dst_stride,
                   reinterpret_cast<const float*>(reinterpret_cast<uint8_t*>(src) + y * src_stride),
                   width * sizeof(float));
        }
        return;
    }

    Eigen::BDCSVD<Eigen::MatrixXf> svd_T(T_mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::BDCSVD<Eigen::MatrixXf> svd_noise(noise, Eigen::ComputeThinU | Eigen::ComputeThinV);

    Eigen::VectorXf S = svd_T.singularValues();
    Eigen::VectorXf S_noise = svd_noise.singularValues();

    auto [beta, sig, ratio] = fit_mp_distribution(S_noise, cutoff);
    MPDistribution mp_dist(beta, sig, ratio);

    for (int i = 0; i < S.size(); i++) {
        double cdf_val = mp_dist.cdf(S(i));
        S(i) = S(i) - S(i) * (1.0 - cdf_val) * (1.0 - tol);
    }

    Eigen::MatrixXf result = svd_T.matrixU() * S.asDiagonal() * svd_T.matrixV().transpose();

    Eigen::Map<Eigen::MatrixXf>(T.data(), height, width) = result;

    for (int i = 0; i < width * height; i++) {
        float lb = std::min(src[i], T[i]);
        float ub = std::max(src[i], T[i]);
        T[i] = std::clamp(ref_ptr[i], lb, ub);
    }

    for (int y = 0; y < height; y++) {
        memcpy(reinterpret_cast<uint8_t*>(dst) + y * dst_stride,
               T.data() + y * width,
               width * sizeof(float));
    }
}

static const VSFrame *VS_CC spannsGetFrame(int n, int activationReason, void *instanceData, 
                                         void **frameData, VSFrameContext *frameCtx, 
                                         VSCore *core, const VSAPI *vsapi) {
    SpannsData *d = (SpannsData *)instanceData;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        if (d->has_ref) vsapi->requestFrameFilter(n, d->ref, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSFrame *ref = d->has_ref ? vsapi->getFrameFilter(n, d->ref, frameCtx) : nullptr;
        
        VSFrame *dst = vsapi->newVideoFrame(vsapi->getVideoFrameFormat(src), 
                                            vsapi->getFrameWidth(src, 0), 
                                            vsapi->getFrameHeight(src, 0), 
                                            src, core);

        for (int p = 0; p < vsapi->getVideoFrameFormat(src)->numPlanes; p++) {
            const float *srcp = (const float *)vsapi->getReadPtr(src, p);
            const float *refp = ref ? (const float *)vsapi->getReadPtr(ref, p) : nullptr;
            float *dstp = (float *)vsapi->getWritePtr(dst, p);

            ptrdiff_t src_stride = vsapi->getStride(src, p);
            ptrdiff_t ref_stride = ref ? vsapi->getStride(ref, p) : 0;
            ptrdiff_t dst_stride = vsapi->getStride(dst, p);
            
            int plane_width = vsapi->getFrameWidth(src, p);
            int plane_height = vsapi->getFrameHeight(src, p);
            
            process_plane_spanns(srcp, src_stride,
                                refp, ref_stride,
                                dstp, dst_stride,
                                plane_width, plane_height,
                                d->sigma, d->tol, d->cutoff);
        }
        
        vsapi->freeFrame(src);
        if (ref) vsapi->freeFrame(ref);
        
        return dst;
    }
    return nullptr;

    (void) frameData;
}

static void VS_CC spannsFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    SpannsData *d = (SpannsData *)instanceData;
    vsapi->freeNode(d->node);
    if (d->ref) vsapi->freeNode(d->ref);
    free(d);

    (void) core;
}

static void VS_CC spannsCreate(const VSMap *in, VSMap *out, void *userData, 
                             VSCore *core, const VSAPI *vsapi) {
    SpannsData d;
    SpannsData *data;
    int err;
    
    d.node = vsapi->mapGetNode(in, "clip", 0, 0);
    d.ref = vsapi->mapGetNode(in, "ref", 0, &err);
    d.has_ref = err != 1;

    const VSVideoInfo *vi = vsapi->getVideoInfo(d.node);

    if(d.has_ref){
        const VSVideoInfo *refi = vsapi->getVideoInfo(d.ref);
        [[unlikely]] if(!vsh::isSameVideoFormat(&vi->format, &refi->format)){
            vsapi->mapSetError(out, "Spanns: clip and ref1 must have same format!");
            vsapi->freeNode(d.node);
            vsapi->freeNode(d.ref);
            return;
        }
    }
    
    [[unlikely]] if (!vsh::isConstantVideoFormat(vi) || vi->format.sampleType != stFloat || 
        vi->format.bitsPerSample != 32 ||
        (vi->format.numPlanes != 1 && vi->format.numPlanes != 3)) {
        vsapi->mapSetError(out, "Spanns: only constant format 32bit float input supported!");
        vsapi->freeNode(d.node);
        if (d.has_ref) vsapi->freeNode(d.ref);
        return;
    }
    
    d.sigma = (float)vsapi->mapGetFloat(in, "sigma", 0, &err);
    if (err) d.sigma = 1.0f;
    [[unlikely]] if (d.sigma < 1.0f) {
        vsapi->mapSetError(out, "Spanns: sigma must be greater or equal to 1.)!");
        vsapi->freeNode(d.node);
        if (d.has_ref) vsapi->freeNode(d.ref);
        return;
    }
    
    d.tol = (float)vsapi->mapGetFloat(in, "tol", 0, &err);
    if (err) d.tol = 0.7f;
    [[unlikely]] if (d.tol > 1 || d.tol < 0) {
        vsapi->mapSetError(out, "Spanns: tol must be a float in range [0, 1]!");
        vsapi->freeNode(d.node);
        if (d.has_ref) vsapi->freeNode(d.ref);
        return;
    }
    
    d.cutoff = (float)vsapi->mapGetFloat(in, "cutoff", 0, &err);
    if (err) d.cutoff = 0.9f;
    [[unlikely]] if (d.cutoff < 0.5) {
        vsapi->mapSetError(out, "Spanns: gamma must be greater than 0.5!");
        vsapi->freeNode(d.node);
        if (d.has_ref) vsapi->freeNode(d.ref);
        return;
    }
    
    data = (SpannsData *)malloc(sizeof(d));
    *data = d;
    
    VSFilterDependency deps[] = {{d.node, rpStrictSpatial}};
    vsapi->createVideoFilter(out, "Spanns", vi, spannsGetFrame, spannsFree, 
                            fmParallel, deps, 1, data, core);

    (void) userData;
}

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin("com.yuygfgg.spanns", "spanns", 
                        "SPANNS Denoising", VS_MAKE_VERSION(1, 0), 
                        VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("SPANNS",
                            "clip:vnode;ref:vnode:opt;"
                            "sigma:float:opt;tol:float:opt;cutoff:float:opt;",
                            "clip:vnode;",
                            spannsCreate, nullptr, plugin);
}
