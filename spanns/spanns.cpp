#include <algorithm>
#include <vector>
#include "VapourSynth4.h"
#include "VSHelper4.h"
#include <boost/sort/pdqsort/pdqsort.hpp>
#include <Eigen/SVD>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <gsl/gsl_multimin.h>

#include "wavelib.h"


typedef struct {
    VSNode *node;
    VSNode *ref1;
    VSNode *ref2;
    float sigma;
    float tol;
    float gamma;
    int passes;
    bool has_ref1;
    bool has_ref2;
} SpannsData;

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
    double sigma = gsl_vector_get(v, 0);
    double ratio = gsl_vector_get(v, 1);
    
    if (sigma <= 0.0 || ratio <= 0.0 || ratio >= 1.0)
        return GSL_POSINF;

    double beta = 1.0;
    MpFitData *d = (MpFitData *)(params);
    const std::vector<double> &x = *(d->x);

    double lambda_minus = beta * sigma * sigma * std::pow(1.0 - std::sqrt(ratio), 2);
    double lambda_plus = beta * sigma * sigma * std::pow(1.0 + std::sqrt(ratio), 2);
    
    double nll = 0.0;
    int valid_points = 0;
    
    for (auto val : x) {
        if (val > lambda_minus && val < lambda_plus) {
            double p = mp_pdf(val, beta, sigma, ratio);
            nll -= std::log(p);
            valid_points++;
        }
    }
    
    if (valid_points < x.size() * 0.5)
        return GSL_POSINF;

    return nll;
}

std::tuple<double, double, double> fit_mp_distribution_gsl(const std::vector<double> &sorted_sv) {
    double ub_init = sorted_sv[size_t(sorted_sv.size() * 0.9)];
    std::vector<double> filtered_sv;
    for (auto s : sorted_sv)
        if (s < ub_init)
            filtered_sv.push_back(s);


    MpFitData data;
    data.x = &filtered_sv;

    gsl_multimin_function f;
    f.f = &nll_function;
    f.n = 2;
    f.params = &data;

    gsl_vector *x_start = gsl_vector_alloc(2);

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

    gsl_vector_set(x_start, 0, sigma_init);
    gsl_vector_set(x_start, 1, ratio_init);

    gsl_multimin_fminimizer *fminimizer = nullptr;
    const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;
    
    fminimizer = gsl_multimin_fminimizer_alloc(T, f.n);
    
    gsl_vector *step_size = gsl_vector_alloc(2);
    gsl_vector_set(step_size, 0, sigma_init * 0.1);
    gsl_vector_set(step_size, 1, 0.1);

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

    double sigma_opt = gsl_vector_get(fminimizer->x, 0);
    double ratio_opt = gsl_vector_get(fminimizer->x, 1);

    gsl_multimin_fminimizer_free(fminimizer);
    gsl_vector_free(x_start);
    gsl_vector_free(step_size);

    return {1.0, sigma_opt, ratio_opt};
}

static std::tuple<double, double, double> fit_mp_distribution(const Eigen::VectorXf& singular_values) {
    std::vector<double> sv(singular_values.data(), singular_values.data() + singular_values.size());
    boost::sort::pdqsort(sv.begin(), sv.end());

    return fit_mp_distribution_gsl(sv);
}

static void median_filter(const float* src, float* dst, int width, int height) {
    cv::Mat src_mat(height, width, CV_32F);
    memcpy(src_mat.data, src, width * height * sizeof(float));
    
    cv::Mat dst_mat;
    cv::medianBlur(src_mat, dst_mat, 3);
    
    memcpy(dst, dst_mat.ptr<float>(), width * height * sizeof(float));
}

static void box_blur(const float* src, float* dst, int width, int height, float sigma) {
    cv::Mat src_mat(height, width, CV_32F);
    cv::Mat temp_mat;
    cv::Mat dst_mat;
    
    memcpy(src_mat.data, src, width * height * sizeof(float));
    
    cv::boxFilter(src_mat, temp_mat, -1, cv::Size(sigma, sigma), 
                cv::Point(-1,-1), true, cv::BORDER_DEFAULT);
    
    cv::boxFilter(temp_mat, dst_mat, -1, cv::Size(sigma, sigma), 
                cv::Point(-1,-1), true, cv::BORDER_DEFAULT);
    
    cv::boxFilter(dst_mat, temp_mat, -1, cv::Size(sigma, sigma), 
                cv::Point(-1,-1), true, cv::BORDER_DEFAULT);
    
    memcpy(dst, temp_mat.ptr<float>(), width * height * sizeof(float));
}

static void process_plane_spanns(const float* src, const float* ref, const float* dref, 
                                float* dst, int width, int height, float sigma, float tol, float gamma, int passes) {
    if (!src || !dst) return;
    
    std::vector<float> temp_ref(width * height);
    std::vector<float> temp_dref(width * height);
    std::vector<float> T(width * height);
    std::vector<float> mask(width * height);
    
    const float* ref_ptr = ref;
    const float* dref_ptr = dref;
    
    if (!ref_ptr) {
        median_filter(src, temp_ref.data(), width, height);
        ref_ptr = temp_ref.data();
    }
    
    if (!dref_ptr) {
        box_blur(src, temp_dref.data(), width, height, sigma);
        dref_ptr = temp_dref.data();
    }
    
    generate_mask(src, width, height, gamma, mask.data());
    
    memcpy(T.data(), src, width * height * sizeof(float));
    
    for (int step = 0; step < passes; step++) {
        Eigen::MatrixXf T_mat = Eigen::Map<const Eigen::MatrixXf>(T.data(), height, width);
        Eigen::MatrixXf ref_mat = Eigen::Map<const Eigen::MatrixXf>(ref_ptr, height, width);

        Eigen::BDCSVD<Eigen::MatrixXf> svd_T(T_mat, Eigen::ComputeThinU | Eigen::ComputeThinV);

        Eigen::MatrixXf noise = ref_mat - T_mat;
        Eigen::BDCSVD<Eigen::MatrixXf> svd_noise(noise, Eigen::ComputeThinU | Eigen::ComputeThinV);

        Eigen::VectorXf S = svd_T.singularValues();
        Eigen::VectorXf S_noise = svd_noise.singularValues();

        auto [beta, sigma, ratio] = fit_mp_distribution(S_noise);
        MPDistribution mp_dist(beta, sigma, ratio);

        for (int i = 0; i < S.size(); i++) {
            double cdf_val = mp_dist.cdf(S(i));
            S(i) = S(i) - S(i) * (1.0 - cdf_val) * (1.0 - tol);
        }

        Eigen::MatrixXf result = svd_T.matrixU() * S.asDiagonal() * svd_T.matrixV().transpose();

        Eigen::Map<Eigen::MatrixXf>(T.data(), height, width) = result;
        
        for (int i = 0; i < width * height; i++) {
            T[i] = mask[i] * T[i] + (1.0f - mask[i]) * dref_ptr[i];
        }
    }
    
    for (int i = 0; i < width * height; i++) {
        float lb = std::min(src[i], ref_ptr[i]);
        float ub = std::max(src[i], ref_ptr[i]);
        dst[i] = std::clamp(T[i], lb, ub);
    }
}

static const VSFrame *VS_CC spannsGetFrame(int n, int activationReason, void *instanceData, 
                                         void **frameData, VSFrameContext *frameCtx, 
                                         VSCore *core, const VSAPI *vsapi) {
    SpannsData *d = (SpannsData *)instanceData;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        if (d->has_ref1) vsapi->requestFrameFilter(n, d->ref1, frameCtx);
        if (d->has_ref2) vsapi->requestFrameFilter(n, d->ref2, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSFrame *ref1 = d->has_ref1 ? vsapi->getFrameFilter(n, d->ref1, frameCtx) : nullptr;
        const VSFrame *ref2 = d->has_ref2 ? vsapi->getFrameFilter(n, d->ref2, frameCtx) : nullptr;
        
        VSFrame *dst = vsapi->newVideoFrame(vsapi->getVideoFrameFormat(src), 
                                            vsapi->getFrameWidth(src, 0), 
                                            vsapi->getFrameHeight(src, 0), 
                                            src, core);
        
        for (int p = 0; p < vsapi->getVideoFrameFormat(src)->numPlanes; p++) {
            const float *srcp = (const float *)vsapi->getReadPtr(src, p);
            const float *ref1p = ref1 ? (const float *)vsapi->getReadPtr(ref1, p) : nullptr;
            const float *ref2p = ref2 ? (const float *)vsapi->getReadPtr(ref2, p) : nullptr;
            float *dstp = (float *)vsapi->getWritePtr(dst, p);
            
            int plane_width = vsapi->getFrameWidth(src, p);
            int plane_height = vsapi->getFrameHeight(src, p);
            
            process_plane_spanns(srcp, ref1p, ref2p, dstp, 
                                plane_width, plane_height,
                                d->sigma, d->tol, d->gamma, d->passes);
        }
        
        vsapi->freeFrame(src);
        if (ref1) vsapi->freeFrame(ref1);
        if (ref2) vsapi->freeFrame(ref2);
        
        return dst;
    }
    return nullptr;

    (void) frameData;
}

static void VS_CC spannsFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    SpannsData *d = (SpannsData *)instanceData;
    vsapi->freeNode(d->node);
    if (d->ref1) vsapi->freeNode(d->ref1);
    if (d->ref2) vsapi->freeNode(d->ref2);
    free(d);

    (void) core;
}

static void VS_CC spannsCreate(const VSMap *in, VSMap *out, void *userData, 
                             VSCore *core, const VSAPI *vsapi) {
    SpannsData d;
    SpannsData *data;
    int err;
    
    d.node = vsapi->mapGetNode(in, "clip", 0, 0);
    d.ref1 = vsapi->mapGetNode(in, "ref1", 0, &err);
    d.has_ref1 = err != 1;
    d.ref2 = vsapi->mapGetNode(in, "ref2", 0, &err);
    d.has_ref2 = err != 1; 

    const VSVideoInfo *vi = vsapi->getVideoInfo(d.node);

    if(d.has_ref1){
        const VSVideoInfo *ref1i = vsapi->getVideoInfo(d.ref1);
        [[unlikely]] if(!vsh::isSameVideoFormat(&vi->format, &ref1i->format)){
            vsapi->mapSetError(out, "Spanns: clip and ref1 must have same format!");
            vsapi->freeNode(d.node);
            vsapi->freeNode(d.ref1);
            return;
        }
    }

    if (d.has_ref2){
        const VSVideoInfo *ref2i = vsapi->getVideoInfo(d.ref2);
        [[unlikely]] if (!vsh::isSameVideoFormat(&vi->format, &ref2i->format)){
            vsapi->mapSetError(out, "Spanns: clip and ref2 must have same format!");
            vsapi->freeNode(d.node);
            vsapi->freeNode(d.ref2);
            return;
        }
    }
    
    [[unlikely]] if (!vsh::isConstantVideoFormat(vi) || vi->format.sampleType != stFloat || 
        vi->format.bitsPerSample != 32 ||
        (vi->format.numPlanes != 1 && vi->format.numPlanes != 3)) {
        vsapi->mapSetError(out, "Spanns: only constant format 32bit float input supported!");
        vsapi->freeNode(d.node);
        if (d.has_ref1) vsapi->freeNode(d.ref1);
        if (d.has_ref2) vsapi->freeNode(d.ref2);
        return;
    }
    
    d.sigma = (float)vsapi->mapGetFloat(in, "sigma", 0, &err);
    if (err) d.sigma = 1.0f;
    
    d.tol = (float)vsapi->mapGetFloat(in, "tol", 0, &err);
    if (err) d.tol = 0.7f;
    [[unlikely]] if (d.tol > 1 || d.tol < 0) {
        vsapi->mapSetError(out, "Spanns: tol must be a float in range [0, 1]!");
        vsapi->freeNode(d.node);
        if (d.has_ref1) vsapi->freeNode(d.ref1);
        if (d.has_ref2) vsapi->freeNode(d.ref2);
        return;
    }
    
    d.gamma = (float)vsapi->mapGetFloat(in, "gamma", 0, &err);
    if (err) d.gamma = 0.5f;
    [[unlikely]] if (d.gamma > 1 || d.gamma < 0) {
        vsapi->mapSetError(out, "Spanns: gamma must be a float in range [0, 1]!");
        vsapi->freeNode(d.node);
        if (d.has_ref1) vsapi->freeNode(d.ref1);
        if (d.has_ref2) vsapi->freeNode(d.ref2);
        return;
    }
    
    d.passes = (int)vsapi->mapGetInt(in, "passes", 0, &err);
    if (err) d.passes = 2;
    [[unlikely]] if (d.passes < 1) {
        vsapi->mapSetError(out, "Spanns: passes must be integer >= 1!");
        vsapi->freeNode(d.node);
        if (d.has_ref1) vsapi->freeNode(d.ref1);
        if (d.has_ref2) vsapi->freeNode(d.ref2);
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
                            "clip:vnode;ref1:vnode:opt;ref2:vnode:opt;"
                            "sigma:float:opt;tol:float:opt;gamma:float:opt;"
                            "passes:int:opt;",
                            "clip:vnode;",
                            spannsCreate, nullptr, plugin);
}