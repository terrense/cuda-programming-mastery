#ifndef AUTO_TUNING_SYSTEM_H
#define AUTO_TUNING_SYSTEM_H

#include "performance_monitor.h"
#include <functional>
#include <map>
#include <vector>
#include <memory>
#include <random>

namespace yolo_acceleration {

// 自动调优系统实现
class AutoTuningSystem {
public:
    // 调优配置
    struct TuningConfig {
        std::vector<int> batch_sizes;           // 批大小候选
        std::vector<std::string> precisions;   // 精度候选 ("fp32", "fp16", "int8")
        std::vector<int> stream_counts;        // 流数量候选
        int max_iterations;                    // 最大迭代次数
        float convergence_threshold;           // 收敛阈值
        std::string optimization_target;       // 优化目标 ("fps", "latency", "throughput")
        bool enable_early_stopping;           // 启用早停
        int patience;                         // 早停耐心值

        TuningConfig();
    };

    // 调优结果
    struct TuningResult {
        int optimal_batch_size;
        std::string optimal_precision;
        int optimal_stream_count;
        RealtimeMetrics best_metrics;
        std::vector<ABTestingFramework::ABTestResult> test_results;
        std::string tuning_summary;
        float total_improvement_percent;
        int total_iterations;
        float tuning_time_minutes;

        TuningResult();
        std::string toString() const;
    };

    // 贝叶斯优化参数
    struct BayesianOptConfig {
        int n_initial_points;      // 初始随机点数量
        int n_calls;              // 总调用次数
        float acquisition_func_kappa; // 采集函数参数
        std::string acquisition_func; // 采集函数类型

        BayesianOptConfig();
    };

private:
    std::unique_ptr<RealtimePerformanceMonitor> monitor_;
    std::unique_ptr<ABTestingFramework> ab_tester_;
    std::unique_ptr<YOLOBottleneckAnalyzer> analyzer_;

    // 历史调优数据
    std::map<std::tuple<int, std::string, int>, RealtimeMetrics> tuning_history_;
    std::mt19937 random_generator_;

public:
    AutoTuningSystem();
    ~AutoTuningSystem();

    // 执行自动调优
    TuningResult autoTune(
        const TuningConfig& config,
        std::function<RealtimeMetrics(int, const std::string&, int)> inference_function);

    // 网格搜索调优
    TuningResult gridSearchTuning(
        const TuningConfig& config,
        std::function<RealtimeMetrics(int, const std::string&, int)> inference_function);

    // 贝叶斯优化调优
    TuningResult bayesianOptimization(
        const TuningConfig& config,
        const BayesianOptConfig& bayes_config,
        std::function<RealtimeMetrics(int, const std::string&, int)> inference_function);

    // 遗传算法调优
    TuningResult geneticAlgorithmTuning(
        const TuningConfig& config,
        std::function<RealtimeMetrics(int, const std::string&, int)> inference_function);

    // 多目标优化
    struct MultiObjectiveResult {
        std::vector<TuningResult> pareto_front;
        std::string analysis_summary;

        std::string toString() const;
    };

    MultiObjectiveResult multiObjectiveOptimization(
        const TuningConfig& config,
        const std::vector<std::string>& objectives,
        std::function<RealtimeMetrics(int, const std::string&, int)> inference_function);

    // 在线自适应调优
    class OnlineAdaptiveTuner {
    private:
        AutoTuningSystem* parent_;
        TuningConfig config_;
        std::function<RealtimeMetrics(int, const std::string&, int)> inference_func_;

        // 当前最优配置
        int current_batch_size_;
        std::string current_precision_;
        int current_stream_count_;

        // 性能监控
        std::vector<RealtimeMetrics> recent_performance_;
        std::chrono::steady_clock::time_point last_tuning_;

        // 自适应参数
        float performance_degradation_threshold_;
        int min_samples_before_retuning_;
        std::chrono::minutes retuning_interval_;

    public:
        OnlineAdaptiveTuner(AutoTuningSystem* parent,
                           const TuningConfig& config,
                           std::function<RealtimeMetrics(int, const std::string&, int)> func);

        void start();
        void stop();
        void recordPerformance(const RealtimeMetrics& metrics);
        bool shouldRetune() const;
        TuningResult performAdaptiveTuning();

        // 获取当前配置
        std::tuple<int, std::string, int> getCurrentConfig() const;
    };

    std::unique_ptr<OnlineAdaptiveTuner> createOnlineAdaptiveTuner(
        const TuningConfig& config,
        std::function<RealtimeMetrics(int, const std::string&, int)> inference_function);

    // 生成调优报告
    std::string generateTuningReport(const TuningResult& result);
    std::string generateComparisonReport(const std::vector<TuningResult>& results);

    // 保存和加载调优历史
    bool saveTuningHistory(const std::string& filename);
    bool loadTuningHistory(const std::string& filename);

    // 获取调优建议
    std::vector<YOLOOptimizationSuggestion> getTuningRecommendations(
        const RealtimeMetrics& current_metrics);

private:
    // 评估配置性能
    float evaluateConfiguration(int batch_size, const std::string& precision,
                               int stream_count, const std::string& target,
                               const RealtimeMetrics& metrics);

    // 生成配置网格
    std::vector<std::tuple<int, std::string, int>> generateConfigurationGrid(
        const TuningConfig& config);

    // 贝叶斯优化辅助函数
    std::tuple<int, std::string, int> selectNextConfiguration(
        const BayesianOptConfig& bayes_config,
        const std::vector<std::tuple<int, std::string, int>>& evaluated_configs,
        const std::vector<float>& scores);

    float calculateAcquisitionFunction(
        const std::tuple<int, std::string, int>& config,
        const std::vector<std::tuple<int, std::string, int>>& evaluated_configs,
        const std::vector<float>& scores,
        const BayesianOptConfig& bayes_config);

    // 遗传算法辅助函数
    struct Individual {
        int batch_size;
        std::string precision;
        int stream_count;
        float fitness;

        Individual();
        Individual(int bs, const std::string& prec, int sc);
    };

    std::vector<Individual> initializePopulation(const TuningConfig& config, int population_size);
    std::vector<Individual> selection(const std::vector<Individual>& population, int select_count);
    std::vector<Individual> crossover(const std::vector<Individual>& parents,
                                     const TuningConfig& config);
    void mutate(std::vector<Individual>& population, const TuningConfig& config, float mutation_rate);
    void evaluatePopulation(std::vector<Individual>& population,
                           std::function<RealtimeMetrics(int, const std::string&, int)> inference_function,
                           const std::string& target);

    // 多目标优化辅助函数
    bool isDominated(const RealtimeMetrics& metrics1, const RealtimeMetrics& metrics2,
                    const std::vector<std::string>& objectives);
    std::vector<TuningResult> calculateParetoFront(const std::vector<TuningResult>& results,
                                                   const std::vector<std::string>& objectives);

    // 统计分析
    float calculateConfidenceInterval(const std::vector<float>& values, float confidence_level);
    bool isStatisticallySignificant(const std::vector<float>& baseline,
                                   const std::vector<float>& variant,
                                   float significance_level);
};

// 性能预测器
class PerformancePredictor {
public:
    // 预测模型类型
    enum class ModelType {
        LINEAR_REGRESSION,
        POLYNOMIAL_REGRESSION,
        NEURAL_NETWORK,
        ENSEMBLE
    };

    // 特征向量
    struct FeatureVector {
        int batch_size;
        float precision_factor;  // fp32=1.0, fp16=0.5, int8=0.25
        int stream_count;
        float gpu_memory_gb;
        int gpu_compute_capability;

        std::vector<float> toVector() const;
    };

    // 预测结果
    struct PredictionResult {
        RealtimeMetrics predicted_metrics;
        float confidence_score;
        std::string model_used;

        std::string toString() const;
    };

private:
    ModelType model_type_;
    std::vector<FeatureVector> training_features_;
    std::vector<RealtimeMetrics> training_targets_;
    bool is_trained_;

public:
    PerformancePredictor(ModelType type = ModelType::LINEAR_REGRESSION);

    // 训练预测模型
    bool train(const std::vector<FeatureVector>& features,
              const std::vector<RealtimeMetrics>& targets);

    // 预测性能
    PredictionResult predict(const FeatureVector& features);

    // 批量预测
    std::vector<PredictionResult> batchPredict(const std::vector<FeatureVector>& features);

    // 模型评估
    float evaluateModel(const std::vector<FeatureVector>& test_features,
                       const std::vector<RealtimeMetrics>& test_targets);

    // 特征重要性分析
    std::map<std::string, float> analyzeFeatureImportance();

    // 保存和加载模型
    bool saveModel(const std::string& filename);
    bool loadModel(const std::string& filename);

private:
    // 线性回归实现
    std::vector<float> trainLinearRegression(const std::vector<std::vector<float>>& X,
                                            const std::vector<float>& y);
    float predictLinearRegression(const std::vector<float>& features,
                                 const std::vector<float>& weights);

    // 多项式回归实现
    std::vector<float> generatePolynomialFeatures(const std::vector<float>& features, int degree);

    // 简单神经网络实现
    class SimpleNeuralNetwork {
    private:
        std::vector<std::vector<float>> weights_;
        std::vector<float> biases_;
        int input_size_, hidden_size_, output_size_;

    public:
        SimpleNeuralNetwork(int input_size, int hidden_size, int output_size);
        void train(const std::vector<std::vector<float>>& X, const std::vector<float>& y,
                  int epochs, float learning_rate);
        float predict(const std::vector<float>& features);

    private:
        float sigmoid(float x);
        float sigmoidDerivative(float x);
    };

    std::unique_ptr<SimpleNeuralNetwork> neural_network_;
};

} // namespace yolo_acceleration

#endif // AUTO_TUNING_SYSTEM_H
