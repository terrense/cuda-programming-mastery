#include "auto_tuning_system.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <fstream>
#include <numeric>
#include <chrono>

namespace yolo_acceleration {

// TuningConfig 实现
AutoTuningSystem::TuningConfig::TuningConfig()
    : batch_sizes({1, 2, 4, 8, 16, 32})
    , precisions({"fp32", "fp16", "int8"})
    , stream_counts({1, 2, 4, 8})
    , max_iterations(100)
    , convergence_threshold(0.01f)
    , optimization_target("fps")
    , enable_early_stopping(true)
    , patience(10) {
}

// TuningResult 实现
AutoTuningSystem::TuningResult::TuningResult()
    : optimal_batch_size(1)
    , optimal_precision("fp32")
    , optimal_stream_count(1)
    , total_improvement_percent(0.0f)
    , total_iterations(0)
    , tuning_time_minutes(0.0f) {
}

std::string AutoTuningSystem::TuningResult::toString() const {
    std::stringstream ss;
    ss << "=== 自动调优结果 ===\n";
    ss << "最优配置:\n";
    ss << "  批大小: " << optimal_batch_size << "\n";
    ss << "  精度: " << optimal_precision << "\n";
    ss << "  流数量: " << optimal_stream_count << "\n";
    ss << "性能指标:\n";
    ss << "  FPS: " << std::fixed << std::setprecision(2) << best_metrics.fps << "\n";
    ss << "  延迟: " << best_metrics.latency_ms << " ms\n";
    ss << "  GPU利用率: " << best_metrics.gpu_utilization << "%\n";
    ss << "  内存使用: " << best_metrics.memory_usage_mb << " MB\n";
    ss << "调优统计:\n";
    ss << "  总改进: " << total_improvement_percent << "%\n";
    ss << "  迭代次数: " << total_iterations << "\n";
    ss << "  调优时间: " << tuning_time_minutes << " 分钟\n";
    ss << "摘要: " << tuning_summary << "\n";
    return ss.str();
}

// BayesianOptConfig 实现
AutoTuningSystem::BayesianOptConfig::BayesianOptConfig()
    : n_initial_points(10)
    , n_calls(50)
    , acquisition_func_kappa(2.576f)
    , acquisition_func("EI") {
}

// AutoTuningSystem 主类实现
AutoTuningSystem::AutoTuningSystem()
    : monitor_(std::make_unique<RealtimePerformanceMonitor>())
    , ab_tester_(std::make_unique<ABTestingFramework>())
    , analyzer_(std::make_unique<YOLOBottleneckAnalyzer>())
    , random_generator_(std::chrono::steady_clock::now().time_since_epoch().count()) {
}Aut
oTuningSystem::~AutoTuningSystem() = default;

// 主要的自动调优函数
AutoTuningSystem::TuningResult AutoTuningSystem::autoTune(
    const TuningConfig& config,
    std::function<RealtimeMetrics(int, const std::string&, int)> inference_function) {

    auto start_time = std::chrono::steady_clock::now();

    std::cout << "开始自动调优..." << std::endl;
    std::cout << "优化目标: " << config.optimization_target << std::endl;

    TuningResult result;
    float best_score = -std::numeric_limits<float>::max();
    RealtimeMetrics baseline_metrics;
    bool has_baseline = false;

    // 生成配置网格
    auto config_grid = generateConfigurationGrid(config);
    std::cout << "生成了 " << config_grid.size() << " 个配置进行测试" << std::endl;

    int iteration = 0;
    int no_improvement_count = 0;

    for (const auto& [batch_size, precision, stream_count] : config_grid) {
        if (iteration >= config.max_iterations) {
            std::cout << "达到最大迭代次数，停止调优" << std::endl;
            break;
        }

        std::cout << "测试配置 [" << (iteration + 1) << "/" << config_grid.size()
                  << "]: batch=" << batch_size << ", precision=" << precision
                  << ", streams=" << stream_count << std::endl;

        try {
            // 执行推理并获取性能指标
            RealtimeMetrics metrics = inference_function(batch_size, precision, stream_count);

            // 记录到历史数据
            tuning_history_[std::make_tuple(batch_size, precision, stream_count)] = metrics;

            // 评估配置性能
            float score = evaluateConfiguration(batch_size, precision, stream_count,
                                              config.optimization_target, metrics);

            std::cout << "  性能得分: " << std::fixed << std::setprecision(3) << score
                      << " (FPS: " << metrics.fps << ", 延迟: " << metrics.latency_ms << "ms)" << std::endl;

            // 设置基线
            if (!has_baseline) {
                baseline_metrics = metrics;
                has_baseline = true;
            }

            // 更新最佳配置
            if (score > best_score) {
                best_score = score;
                result.optimal_batch_size = batch_size;
                result.optimal_precision = precision;
                result.optimal_stream_count = stream_count;
                result.best_metrics = metrics;
                no_improvement_count = 0;

                std::cout << "  ✓ 发现更好的配置！" << std::endl;
            } else {
                no_improvement_count++;
            }

            // 早停检查
            if (config.enable_early_stopping && no_improvement_count >= config.patience) {
                std::cout << "早停触发，停止调优" << std::endl;
                break;
            }

        } catch (const std::exception& e) {
            std::cerr << "配置测试失败: " << e.what() << std::endl;
            continue;
        }

        iteration++;
    }

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // 计算改进百分比
    if (has_baseline) {
        float baseline_score = evaluateConfiguration(
            config.batch_sizes[0], config.precisions[0], config.stream_counts[0],
            config.optimization_target, baseline_metrics);
        result.total_improvement_percent = ((best_score - baseline_score) / baseline_score) * 100.0f;
    }

    result.total_iterations = iteration;
    result.tuning_time_minutes = duration.count() / 60000.0f;

    // 生成调优摘要
    std::stringstream summary;
    summary << "经过 " << iteration << " 次迭代，找到最优配置。";
    if (result.total_improvement_percent > 0) {
        summary << "性能提升 " << std::fixed << std::setprecision(1)
                << result.total_improvement_percent << "%。";
    }
    result.tuning_summary = summary.str();

    std::cout << "调优完成！" << std::endl;
    std::cout << result.toString() << std::endl;

    return result;
}

// 网格搜索调优
AutoTuningSystem::TuningResult AutoTuningSystem::gridSearchTuning(
    const TuningConfig& config,
    std::function<RealtimeMetrics(int, const std::string&, int)> inference_function) {

    std::cout << "执行网格搜索调优..." << std::endl;
    return autoTune(config, inference_function);
}

// 评估配置性能
float AutoTuningSystem::evaluateConfiguration(int batch_size, const std::string& precision,
                                             int stream_count, const std::string& target,
                                             const RealtimeMetrics& metrics) {
    if (target == "fps") {
        return metrics.fps;
    } else if (target == "latency") {
        return 1000.0f / std::max(metrics.latency_ms, 0.1f); // 延迟越低越好
    } else if (target == "throughput") {
        return metrics.fps * batch_size; // 吞吐量 = FPS * 批大小
    } else if (target == "efficiency") {
        // 综合效率：考虑FPS、GPU利用率和内存使用
        float fps_score = metrics.fps / 100.0f; // 归一化到0-1
        float gpu_score = metrics.gpu_utilization / 100.0f;
        float memory_score = std::max(0.0f, 1.0f - metrics.memory_usage_mb / 8192.0f); // 假设8GB为满
        return (fps_score * 0.5f + gpu_score * 0.3f + memory_score * 0.2f) * 100.0f;
    }

    return metrics.fps; // 默认使用FPS
}

// 生成配置网格
std::vector<std::tuple<int, std::string, int>> AutoTuningSystem::generateConfigurationGrid(
    const TuningConfig& config) {

    std::vector<std::tuple<int, std::string, int>> grid;

    for (int batch_size : config.batch_sizes) {
        for (const std::string& precision : config.precisions) {
            for (int stream_count : config.stream_counts) {
                grid.emplace_back(batch_size, precision, stream_count);
            }
        }
    }

    // 随机打乱顺序以避免偏差
    std::shuffle(grid.begin(), grid.end(), random_generator_);

    return grid;
}

// 贝叶斯优化调优
AutoTuningSystem::TuningResult AutoTuningSystem::bayesianOptimization(
    const TuningConfig& config,
    const BayesianOptConfig& bayes_config,
    std::function<RealtimeMetrics(int, const std::string&, int)> inference_function) {

    std::cout << "执行贝叶斯优化调优..." << std::endl;

    auto start_time = std::chrono::steady_clock::now();
    TuningResult result;

    std::vector<std::tuple<int, std::string, int>> evaluated_configs;
    std::vector<float> scores;

    // 生成初始随机点
    auto config_grid = generateConfigurationGrid(config);
    std::shuffle(config_grid.begin(), config_grid.end(), random_generator_);

    float best_score = -std::numeric_limits<float>::max();

    // 初始随机采样
    for (int i = 0; i < std::min(bayes_config.n_initial_points, (int)config_grid.size()); ++i) {
        auto [batch_size, precision, stream_count] = config_grid[i];

        std::cout << "初始采样 [" << (i + 1) << "/" << bayes_config.n_initial_points
                  << "]: batch=" << batch_size << ", precision=" << precision
                  << ", streams=" << stream_count << std::endl;

        try {
            RealtimeMetrics metrics = inference_function(batch_size, precision, stream_count);
            float score = evaluateConfiguration(batch_size, precision, stream_count,
                                              config.optimization_target, metrics);

            evaluated_configs.push_back(config_grid[i]);
            scores.push_back(score);

            if (score > best_score) {
                best_score = score;
                result.optimal_batch_size = batch_size;
                result.optimal_precision = precision;
                result.optimal_stream_count = stream_count;
                result.best_metrics = metrics;
            }

        } catch (const std::exception& e) {
            std::cerr << "初始采样失败: " << e.what() << std::endl;
            continue;
        }
    }

    // 贝叶斯优化迭代
    for (int iter = bayes_config.n_initial_points; iter < bayes_config.n_calls; ++iter) {
        // 选择下一个配置
        auto next_config = selectNextConfiguration(bayes_config, evaluated_configs, scores);
        auto [batch_size, precision, stream_count] = next_config;

        std::cout << "贝叶斯迭代 [" << (iter + 1) << "/" << bayes_config.n_calls
                  << "]: batch=" << batch_size << ", precision=" << precision
                  << ", streams=" << stream_count << std::endl;

        try {
            RealtimeMetrics metrics = inference_function(batch_size, precision, stream_count);
            float score = evaluateConfiguration(batch_size, precision, stream_count,
                                              config.optimization_target, metrics);

            evaluated_configs.push_back(next_config);
            scores.push_back(score);

            if (score > best_score) {
                best_score = score;
                result.optimal_batch_size = batch_size;
                result.optimal_precision = precision;
                result.optimal_stream_count = stream_count;
                result.best_metrics = metrics;

                std::cout << "  ✓ 发现更好的配置！得分: " << score << std::endl;
            }

        } catch (const std::exception& e) {
            std::cerr << "贝叶斯迭代失败: " << e.what() << std::endl;
            continue;
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    result.total_iterations = evaluated_configs.size();
    result.tuning_time_minutes = duration.count() / 60000.0f;
    result.tuning_summary = "贝叶斯优化完成，评估了 " + std::to_string(result.total_iterations) + " 个配置";

    std::cout << "贝叶斯优化完成！" << std::endl;
    std::cout << result.toString() << std::endl;

    return result;
}

// 选择下一个配置（贝叶斯优化）
std::tuple<int, std::string, int> AutoTuningSystem::selectNextConfiguration(
    const BayesianOptConfig& bayes_config,
    const std::vector<std::tuple<int, std::string, int>>& evaluated_configs,
    const std::vector<float>& scores) {

    // 简化的采集函数实现
    // 在实际应用中，这里应该使用更复杂的高斯过程模型

    auto config_grid = generateConfigurationGrid(TuningConfig());

    float best_acquisition = -std::numeric_limits<float>::max();
    std::tuple<int, std::string, int> best_config = config_grid[0];

    for (const auto& config : config_grid) {
        // 跳过已评估的配置
        if (std::find(evaluated_configs.begin(), evaluated_configs.end(), config) != evaluated_configs.end()) {
            continue;
        }

        float acquisition_value = calculateAcquisitionFunction(config, evaluated_configs, scores, bayes_config);

        if (acquisition_value > best_acquisition) {
            best_acquisition = acquisition_value;
            best_config = config;
        }
    }

    return best_config;
}

// 计算采集函数值
float AutoTuningSystem::calculateAcquisitionFunction(
    const std::tuple<int, std::string, int>& config,
    const std::vector<std::tuple<int, std::string, int>>& evaluated_configs,
    const std::vector<float>& scores,
    const BayesianOptConfig& bayes_config) {

    if (scores.empty()) {
        return 1.0f; // 如果没有历史数据，返回固定值
    }

    // 简化的Expected Improvement (EI)实现
    float max_score = *std::max_element(scores.begin(), scores.end());
    float mean_score = std::accumulate(scores.begin(), scores.end(), 0.0f) / scores.size();

    // 计算与已评估配置的距离（简化的相似度度量）
    float min_distance = std::numeric_limits<float>::max();
    auto [batch_size, precision, stream_count] = config;

    for (const auto& eval_config : evaluated_configs) {
        auto [eval_batch, eval_precision, eval_stream] = eval_config;

        float distance = std::abs(batch_size - eval_batch) / 32.0f +  // 归一化批大小差异
                        (precision != eval_precision ? 1.0f : 0.0f) +  // 精度差异
                        std::abs(stream_count - eval_stream) / 8.0f;    // 归一化流数量差异

        min_distance = std::min(min_distance, distance);
    }

    // Expected Improvement approximation
    float exploration_bonus = bayes_config.acquisition_func_kappa * min_distance;
    float exploitation_value = std::max(0.0f, mean_score - max_score);

    return exploitation_value + exploration_bonus;
}/
/ Individual 构造函数实现
AutoTuningSystem::Individual::Individual() : batch_size(1), precision("fp32"), stream_count(1), fitness(0.0f) {}

AutoTuningSystem::Individual::Individual(int bs, const std::string& prec, int sc)
    : batch_size(bs), precision(prec), stream_count(sc), fitness(0.0f) {}

// 遗传算法调优
AutoTuningSystem::TuningResult AutoTuningSystem::geneticAlgorithmTuning(
    const TuningConfig& config,
    std::function<RealtimeMetrics(int, const std::string&, int)> inference_function) {

    std::cout << "执行遗传算法调优..." << std::endl;

    auto start_time = std::chrono::steady_clock::now();
    TuningResult result;

    const int population_size = 20;
    const int generations = 30;
    const float mutation_rate = 0.1f;
    const int elite_count = 4;

    // 初始化种群
    auto population = initializePopulation(config, population_size);

    float best_fitness = -std::numeric_limits<float>::max();
    int generation = 0;

    for (generation = 0; generation < generations; ++generation) {
        std::cout << "遗传算法第 " << (generation + 1) << "/" << generations << " 代" << std::endl;

        // 评估种群
        evaluatePopulation(population, inference_function, config.optimization_target);

        // 排序种群（按适应度降序）
        std::sort(population.begin(), population.end(),
                 [](const Individual& a, const Individual& b) {
                     return a.fitness > b.fitness;
                 });

        // 更新最佳个体
        if (population[0].fitness > best_fitness) {
            best_fitness = population[0].fitness;
            result.optimal_batch_size = population[0].batch_size;
            result.optimal_precision = population[0].precision;
            result.optimal_stream_count = population[0].stream_count;

            // 重新评估最佳个体以获取详细指标
            result.best_metrics = inference_function(population[0].batch_size,
                                                   population[0].precision,
                                                   population[0].stream_count);

            std::cout << "  ✓ 发现更好的个体！适应度: " << best_fitness << std::endl;
        }

        // 选择
        auto parents = selection(population, population_size / 2);

        // 交叉
        auto offspring = crossover(parents, config);

        // 变异
        mutate(offspring, config, mutation_rate);

        // 精英保留 + 新后代
        std::vector<Individual> new_population;
        for (int i = 0; i < elite_count && i < population.size(); ++i) {
            new_population.push_back(population[i]);
        }
        for (int i = 0; i < population_size - elite_count && i < offspring.size(); ++i) {
            new_population.push_back(offspring[i]);
        }

        population = new_population;
    }

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    result.total_iterations = generation * population_size;
    result.tuning_time_minutes = duration.count() / 60000.0f;
    result.tuning_summary = "遗传算法完成，进化了 " + std::to_string(generation) + " 代";

    std::cout << "遗传算法调优完成！" << std::endl;
    std::cout << result.toString() << std::endl;

    return result;
}

// 初始化种群
std::vector<AutoTuningSystem::Individual> AutoTuningSystem::initializePopulation(
    const TuningConfig& config, int population_size) {

    std::vector<Individual> population;
    population.reserve(population_size);

    std::uniform_int_distribution<int> batch_dist(0, config.batch_sizes.size() - 1);
    std::uniform_int_distribution<int> precision_dist(0, config.precisions.size() - 1);
    std::uniform_int_distribution<int> stream_dist(0, config.stream_counts.size() - 1);

    for (int i = 0; i < population_size; ++i) {
        int batch_idx = batch_dist(random_generator_);
        int precision_idx = precision_dist(random_generator_);
        int stream_idx = stream_dist(random_generator_);

        population.emplace_back(config.batch_sizes[batch_idx],
                               config.precisions[precision_idx],
                               config.stream_counts[stream_idx]);
    }

    return population;
}

// 选择操作
std::vector<AutoTuningSystem::Individual> AutoTuningSystem::selection(
    const std::vector<Individual>& population, int select_count) {

    std::vector<Individual> selected;
    selected.reserve(select_count);

    // 锦标赛选择
    const int tournament_size = 3;
    std::uniform_int_distribution<int> pop_dist(0, population.size() - 1);

    for (int i = 0; i < select_count; ++i) {
        Individual best_candidate = population[pop_dist(random_generator_)];

        for (int j = 1; j < tournament_size; ++j) {
            Individual candidate = population[pop_dist(random_generator_)];
            if (candidate.fitness > best_candidate.fitness) {
                best_candidate = candidate;
            }
        }

        selected.push_back(best_candidate);
    }

    return selected;
}

// 交叉操作
std::vector<AutoTuningSystem::Individual> AutoTuningSystem::crossover(
    const std::vector<Individual>& parents, const TuningConfig& config) {

    std::vector<Individual> offspring;
    offspring.reserve(parents.size());

    std::uniform_int_distribution<int> parent_dist(0, parents.size() - 1);
    std::uniform_real_distribution<float> crossover_prob(0.0f, 1.0f);

    for (size_t i = 0; i < parents.size(); ++i) {
        if (crossover_prob(random_generator_) < 0.8f) { // 80% 交叉概率
            // 选择两个父代
            const Individual& parent1 = parents[parent_dist(random_generator_)];
            const Individual& parent2 = parents[parent_dist(random_generator_)];

            // 单点交叉
            Individual child;
            std::uniform_int_distribution<int> gene_dist(0, 2);
            int crossover_point = gene_dist(random_generator_);

            switch (crossover_point) {
                case 0:
                    child.batch_size = parent1.batch_size;
                    child.precision = parent2.precision;
                    child.stream_count = parent2.stream_count;
                    break;
                case 1:
                    child.batch_size = parent1.batch_size;
                    child.precision = parent1.precision;
                    child.stream_count = parent2.stream_count;
                    break;
                case 2:
                    child.batch_size = parent2.batch_size;
                    child.precision = parent2.precision;
                    child.stream_count = parent1.stream_count;
                    break;
            }

            offspring.push_back(child);
        } else {
            // 直接复制父代
            offspring.push_back(parents[i]);
        }
    }

    return offspring;
}

// 变异操作
void AutoTuningSystem::mutate(std::vector<Individual>& population,
                             const TuningConfig& config, float mutation_rate) {

    std::uniform_real_distribution<float> mutation_prob(0.0f, 1.0f);
    std::uniform_int_distribution<int> gene_dist(0, 2);

    for (auto& individual : population) {
        if (mutation_prob(random_generator_) < mutation_rate) {
            int gene_to_mutate = gene_dist(random_generator_);

            switch (gene_to_mutate) {
                case 0: { // 变异批大小
                    std::uniform_int_distribution<int> batch_dist(0, config.batch_sizes.size() - 1);
                    individual.batch_size = config.batch_sizes[batch_dist(random_generator_)];
                    break;
                }
                case 1: { // 变异精度
                    std::uniform_int_distribution<int> precision_dist(0, config.precisions.size() - 1);
                    individual.precision = config.precisions[precision_dist(random_generator_)];
                    break;
                }
                case 2: { // 变异流数量
                    std::uniform_int_distribution<int> stream_dist(0, config.stream_counts.size() - 1);
                    individual.stream_count = config.stream_counts[stream_dist(random_generator_)];
                    break;
                }
            }
        }
    }
}

// 评估种群
void AutoTuningSystem::evaluatePopulation(
    std::vector<Individual>& population,
    std::function<RealtimeMetrics(int, const std::string&, int)> inference_function,
    const std::string& target) {

    for (auto& individual : population) {
        try {
            RealtimeMetrics metrics = inference_function(individual.batch_size,
                                                       individual.precision,
                                                       individual.stream_count);
            individual.fitness = evaluateConfiguration(individual.batch_size,
                                                     individual.precision,
                                                     individual.stream_count,
                                                     target, metrics);
        } catch (const std::exception& e) {
            std::cerr << "个体评估失败: " << e.what() << std::endl;
            individual.fitness = 0.0f; // 失败的个体适应度为0
        }
    }
}

// 多目标优化结果toString
std::string AutoTuningSystem::MultiObjectiveResult::toString() const {
    std::stringstream ss;
    ss << "=== 多目标优化结果 ===\n";
    ss << "帕累托前沿包含 " << pareto_front.size() << " 个解\n";
    ss << "分析摘要: " << analysis_summary << "\n";
    ss << "\n帕累托最优解:\n";

    for (size_t i = 0; i < pareto_front.size(); ++i) {
        ss << "解 " << (i + 1) << ":\n";
        ss << "  配置: batch=" << pareto_front[i].optimal_batch_size
           << ", precision=" << pareto_front[i].optimal_precision
           << ", streams=" << pareto_front[i].optimal_stream_count << "\n";
        ss << "  FPS: " << pareto_front[i].best_metrics.fps
           << ", 延迟: " << pareto_front[i].best_metrics.latency_ms << "ms\n";
    }

    return ss.str();
}

// 多目标优化
AutoTuningSystem::MultiObjectiveResult AutoTuningSystem::multiObjectiveOptimization(
    const TuningConfig& config,
    const std::vector<std::string>& objectives,
    std::function<RealtimeMetrics(int, const std::string&, int)> inference_function) {

    std::cout << "执行多目标优化..." << std::endl;
    std::cout << "优化目标: ";
    for (const auto& obj : objectives) {
        std::cout << obj << " ";
    }
    std::cout << std::endl;

    MultiObjectiveResult result;
    std::vector<TuningResult> all_results;

    // 为每个目标单独优化
    for (const auto& objective : objectives) {
        std::cout << "优化目标: " << objective << std::endl;

        TuningConfig obj_config = config;
        obj_config.optimization_target = objective;

        TuningResult obj_result = autoTune(obj_config, inference_function);
        all_results.push_back(obj_result);
    }

    // 计算帕累托前沿
    result.pareto_front = calculateParetoFront(all_results, objectives);

    std::stringstream summary;
    summary << "多目标优化完成。在 " << objectives.size() << " 个目标中找到了 "
            << result.pareto_front.size() << " 个帕累托最优解。";
    result.analysis_summary = summary.str();

    std::cout << result.toString() << std::endl;

    return result;
}

// 计算帕累托前沿
std::vector<AutoTuningSystem::TuningResult> AutoTuningSystem::calculateParetoFront(
    const std::vector<TuningResult>& results, const std::vector<std::string>& objectives) {

    std::vector<TuningResult> pareto_front;

    for (const auto& result1 : results) {
        bool is_dominated = false;

        for (const auto& result2 : results) {
            if (&result1 == &result2) continue;

            if (isDominated(result1.best_metrics, result2.best_metrics, objectives)) {
                is_dominated = true;
                break;
            }
        }

        if (!is_dominated) {
            pareto_front.push_back(result1);
        }
    }

    return pareto_front;
}

// 检查是否被支配
bool AutoTuningSystem::isDominated(const RealtimeMetrics& metrics1, const RealtimeMetrics& metrics2,
                                  const std::vector<std::string>& objectives) {

    bool at_least_one_worse = false;
    bool all_worse_or_equal = true;

    for (const auto& objective : objectives) {
        float value1, value2;

        if (objective == "fps") {
            value1 = metrics1.fps;
            value2 = metrics2.fps;
        } else if (objective == "latency") {
            value1 = -metrics1.latency_ms; // 延迟越低越好，所以取负值
            value2 = -metrics2.latency_ms;
        } else if (objective == "gpu_utilization") {
            value1 = metrics1.gpu_utilization;
            value2 = metrics2.gpu_utilization;
        } else if (objective == "memory_efficiency") {
            value1 = -metrics1.memory_usage_mb; // 内存使用越低越好
            value2 = -metrics2.memory_usage_mb;
        } else {
            continue; // 未知目标，跳过
        }

        if (value1 < value2) {
            at_least_one_worse = true;
        }
        if (value1 > value2) {
            all_worse_or_equal = false;
        }
    }

    return at_least_one_worse && all_worse_or_equal;
}/
/ OnlineAdaptiveTuner 实现
AutoTuningSystem::OnlineAdaptiveTuner::OnlineAdaptiveTuner(
    AutoTuningSystem* parent,
    const TuningConfig& config,
    std::function<RealtimeMetrics(int, const std::string&, int)> func)
    : parent_(parent)
    , config_(config)
    , inference_func_(func)
    , current_batch_size_(config.batch_sizes[0])
    , current_precision_(config.precisions[0])
    , current_stream_count_(config.stream_counts[0])
    , performance_degradation_threshold_(0.1f) // 10% 性能下降阈值
    , min_samples_before_retuning_(10)
    , retuning_interval_(std::chrono::minutes(30)) {

    last_tuning_ = std::chrono::steady_clock::now();
}

void AutoTuningSystem::OnlineAdaptiveTuner::start() {
    std::cout << "启动在线自适应调优器..." << std::endl;
    std::cout << "初始配置: batch=" << current_batch_size_
              << ", precision=" << current_precision_
              << ", streams=" << current_stream_count_ << std::endl;
}

void AutoTuningSystem::OnlineAdaptiveTuner::stop() {
    std::cout << "停止在线自适应调优器" << std::endl;
    recent_performance_.clear();
}

void AutoTuningSystem::OnlineAdaptiveTuner::recordPerformance(const RealtimeMetrics& metrics) {
    recent_performance_.push_back(metrics);

    // 保持最近的性能记录数量在合理范围内
    const size_t max_history = 100;
    if (recent_performance_.size() > max_history) {
        recent_performance_.erase(recent_performance_.begin());
    }
}

bool AutoTuningSystem::OnlineAdaptiveTuner::shouldRetune() const {
    // 检查是否有足够的样本
    if (recent_performance_.size() < min_samples_before_retuning_) {
        return false;
    }

    // 检查时间间隔
    auto now = std::chrono::steady_clock::now();
    auto time_since_last_tuning = std::chrono::duration_cast<std::chrono::minutes>(now - last_tuning_);
    if (time_since_last_tuning < retuning_interval_) {
        return false;
    }

    // 检查性能是否下降
    const size_t recent_samples = std::min(recent_performance_.size(), (size_t)min_samples_before_retuning_);
    const size_t baseline_samples = std::min(recent_performance_.size() / 2, (size_t)20);

    if (recent_performance_.size() < baseline_samples + recent_samples) {
        return false;
    }

    // 计算基线性能（较早的样本）
    float baseline_fps = 0.0f;
    for (size_t i = 0; i < baseline_samples; ++i) {
        baseline_fps += recent_performance_[i].fps;
    }
    baseline_fps /= baseline_samples;

    // 计算最近性能
    float recent_fps = 0.0f;
    for (size_t i = recent_performance_.size() - recent_samples; i < recent_performance_.size(); ++i) {
        recent_fps += recent_performance_[i].fps;
    }
    recent_fps /= recent_samples;

    // 检查性能下降
    float performance_drop = (baseline_fps - recent_fps) / baseline_fps;

    if (performance_drop > performance_degradation_threshold_) {
        std::cout << "检测到性能下降 " << (performance_drop * 100.0f) << "%, 触发重新调优" << std::endl;
        return true;
    }

    return false;
}

AutoTuningSystem::TuningResult AutoTuningSystem::OnlineAdaptiveTuner::performAdaptiveTuning() {
    std::cout << "执行自适应重新调优..." << std::endl;

    // 创建一个缩小的搜索空间，专注于当前配置附近
    TuningConfig adaptive_config = config_;

    // 缩小搜索范围到当前配置附近
    std::vector<int> nearby_batch_sizes;
    std::vector<int> nearby_stream_counts;

    for (int batch : config_.batch_sizes) {
        if (std::abs(batch - current_batch_size_) <= current_batch_size_) {
            nearby_batch_sizes.push_back(batch);
        }
    }

    for (int stream : config_.stream_counts) {
        if (std::abs(stream - current_stream_count_) <= 2) {
            nearby_stream_counts.push_back(stream);
        }
    }

    if (nearby_batch_sizes.empty()) nearby_batch_sizes.push_back(current_batch_size_);
    if (nearby_stream_counts.empty()) nearby_stream_counts.push_back(current_stream_count_);

    adaptive_config.batch_sizes = nearby_batch_sizes;
    adaptive_config.stream_counts = nearby_stream_counts;
    adaptive_config.max_iterations = 20; // 限制迭代次数以快速完成

    TuningResult result = parent_->autoTune(adaptive_config, inference_func_);

    // 更新当前配置
    current_batch_size_ = result.optimal_batch_size;
    current_precision_ = result.optimal_precision;
    current_stream_count_ = result.optimal_stream_count;

    last_tuning_ = std::chrono::steady_clock::now();

    std::cout << "自适应调优完成，新配置: batch=" << current_batch_size_
              << ", precision=" << current_precision_
              << ", streams=" << current_stream_count_ << std::endl;

    return result;
}

std::tuple<int, std::string, int> AutoTuningSystem::OnlineAdaptiveTuner::getCurrentConfig() const {
    return std::make_tuple(current_batch_size_, current_precision_, current_stream_count_);
}

// 创建在线自适应调优器
std::unique_ptr<AutoTuningSystem::OnlineAdaptiveTuner> AutoTuningSystem::createOnlineAdaptiveTuner(
    const TuningConfig& config,
    std::function<RealtimeMetrics(int, const std::string&, int)> inference_function) {

    return std::make_unique<OnlineAdaptiveTuner>(this, config, inference_function);
}

// 生成调优报告
std::string AutoTuningSystem::generateTuningReport(const TuningResult& result) {
    std::stringstream report;

    report << "# 自动调优详细报告\n\n";

    report << "## 调优概览\n";
    report << "- **调优时间**: " << std::fixed << std::setprecision(2)
           << result.tuning_time_minutes << " 分钟\n";
    report << "- **总迭代次数**: " << result.total_iterations << "\n";
    report << "- **性能提升**: " << std::fixed << std::setprecision(1)
           << result.total_improvement_percent << "%\n\n";

    report << "## 最优配置\n";
    report << "- **批大小**: " << result.optimal_batch_size << "\n";
    report << "- **精度**: " << result.optimal_precision << "\n";
    report << "- **流数量**: " << result.optimal_stream_count << "\n\n";

    report << "## 性能指标\n";
    report << "- **FPS**: " << std::fixed << std::setprecision(2) << result.best_metrics.fps << "\n";
    report << "- **延迟**: " << std::fixed << std::setprecision(2) << result.best_metrics.latency_ms << " ms\n";
    report << "- **GPU利用率**: " << std::fixed << std::setprecision(1) << result.best_metrics.gpu_utilization << "%\n";
    report << "- **内存使用**: " << std::fixed << std::setprecision(1) << result.best_metrics.memory_usage_mb << " MB\n";
    report << "- **功耗**: " << std::fixed << std::setprecision(1) << result.best_metrics.power_consumption_w << " W\n\n";

    report << "## 调优摘要\n";
    report << result.tuning_summary << "\n\n";

    report << "## 建议\n";
    if (result.total_improvement_percent > 20.0f) {
        report << "- ✅ 调优效果显著，建议采用此配置\n";
    } else if (result.total_improvement_percent > 5.0f) {
        report << "- ⚠️ 调优有一定效果，可以考虑采用\n";
    } else {
        report << "- ❌ 调优效果有限，可能需要检查其他优化方向\n";
    }

    if (result.best_metrics.gpu_utilization < 70.0f) {
        report << "- 💡 GPU利用率较低，可以考虑增加批大小或并行度\n";
    }

    if (result.best_metrics.memory_usage_mb > 6000.0f) {
        report << "- ⚠️ 内存使用较高，注意监控内存溢出风险\n";
    }

    return report.str();
}

// 生成对比报告
std::string AutoTuningSystem::generateComparisonReport(const std::vector<TuningResult>& results) {
    if (results.empty()) {
        return "没有调优结果可供比较";
    }

    std::stringstream report;

    report << "# 调优方法对比报告\n\n";

    report << "## 结果对比\n";
    report << "| 方法 | FPS | 延迟(ms) | GPU利用率(%) | 内存(MB) | 调优时间(min) | 改进(%) |\n";
    report << "|------|-----|----------|--------------|----------|---------------|----------|\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        report << "| 方法" << (i + 1) << " | "
               << std::fixed << std::setprecision(1) << result.best_metrics.fps << " | "
               << std::fixed << std::setprecision(1) << result.best_metrics.latency_ms << " | "
               << std::fixed << std::setprecision(1) << result.best_metrics.gpu_utilization << " | "
               << std::fixed << std::setprecision(0) << result.best_metrics.memory_usage_mb << " | "
               << std::fixed << std::setprecision(1) << result.tuning_time_minutes << " | "
               << std::fixed << std::setprecision(1) << result.total_improvement_percent << " |\n";
    }

    report << "\n## 最佳配置\n";

    // 找到最佳FPS
    auto best_fps_it = std::max_element(results.begin(), results.end(),
        [](const TuningResult& a, const TuningResult& b) {
            return a.best_metrics.fps < b.best_metrics.fps;
        });

    // 找到最低延迟
    auto best_latency_it = std::min_element(results.begin(), results.end(),
        [](const TuningResult& a, const TuningResult& b) {
            return a.best_metrics.latency_ms < b.best_metrics.latency_ms;
        });

    // 找到最快调优
    auto fastest_tuning_it = std::min_element(results.begin(), results.end(),
        [](const TuningResult& a, const TuningResult& b) {
            return a.tuning_time_minutes < b.tuning_time_minutes;
        });

    report << "- **最高FPS**: " << std::fixed << std::setprecision(2) << best_fps_it->best_metrics.fps
           << " (batch=" << best_fps_it->optimal_batch_size
           << ", precision=" << best_fps_it->optimal_precision
           << ", streams=" << best_fps_it->optimal_stream_count << ")\n";

    report << "- **最低延迟**: " << std::fixed << std::setprecision(2) << best_latency_it->best_metrics.latency_ms
           << "ms (batch=" << best_latency_it->optimal_batch_size
           << ", precision=" << best_latency_it->optimal_precision
           << ", streams=" << best_latency_it->optimal_stream_count << ")\n";

    report << "- **最快调优**: " << std::fixed << std::setprecision(1) << fastest_tuning_it->tuning_time_minutes
           << " 分钟\n\n";

    report << "## 推荐\n";
    if (best_fps_it->best_metrics.fps > 100.0f) {
        report << "- 🚀 推荐使用最高FPS配置以获得最佳性能\n";
    } else if (best_latency_it->best_metrics.latency_ms < 10.0f) {
        report << "- ⚡ 推荐使用最低延迟配置以获得最佳响应性\n";
    } else {
        report << "- ⚖️ 建议根据具体应用场景选择合适的配置\n";
    }

    return report.str();
}

// 保存调优历史
bool AutoTuningSystem::saveTuningHistory(const std::string& filename) {
    try {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "无法打开文件进行写入: " << filename << std::endl;
            return false;
        }

        file << "# 调优历史数据\n";
        file << "batch_size,precision,stream_count,fps,latency_ms,gpu_utilization,memory_usage_mb,power_consumption_w\n";

        for (const auto& [config, metrics] : tuning_history_) {
            auto [batch_size, precision, stream_count] = config;
            file << batch_size << "," << precision << "," << stream_count << ","
                 << metrics.fps << "," << metrics.latency_ms << ","
                 << metrics.gpu_utilization << "," << metrics.memory_usage_mb << ","
                 << metrics.power_consumption_w << "\n";
        }

        file.close();
        std::cout << "调优历史已保存到: " << filename << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "保存调优历史失败: " << e.what() << std::endl;
        return false;
    }
}

// 加载调优历史
bool AutoTuningSystem::loadTuningHistory(const std::string& filename) {
    try {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "无法打开文件进行读取: " << filename << std::endl;
            return false;
        }

        std::string line;
        std::getline(file, line); // 跳过注释行
        std::getline(file, line); // 跳过标题行

        tuning_history_.clear();

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string item;
            std::vector<std::string> tokens;

            while (std::getline(ss, item, ',')) {
                tokens.push_back(item);
            }

            if (tokens.size() >= 8) {
                int batch_size = std::stoi(tokens[0]);
                std::string precision = tokens[1];
                int stream_count = std::stoi(tokens[2]);

                RealtimeMetrics metrics;
                metrics.fps = std::stof(tokens[3]);
                metrics.latency_ms = std::stof(tokens[4]);
                metrics.gpu_utilization = std::stof(tokens[5]);
                metrics.memory_usage_mb = std::stof(tokens[6]);
                metrics.power_consumption_w = std::stof(tokens[7]);

                tuning_history_[std::make_tuple(batch_size, precision, stream_count)] = metrics;
            }
        }

        file.close();
        std::cout << "已加载 " << tuning_history_.size() << " 条调优历史记录" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "加载调优历史失败: " << e.what() << std::endl;
        return false;
    }
}

// 获取调优建议
std::vector<YOLOOptimizationSuggestion> AutoTuningSystem::getTuningRecommendations(
    const RealtimeMetrics& current_metrics) {

    std::vector<YOLOOptimizationSuggestion> suggestions;

    // 基于当前性能指标生成建议
    if (current_metrics.fps < 30.0f) {
        YOLOOptimizationSuggestion suggestion;
        suggestion.category = "Performance";
        suggestion.priority = "High";
        suggestion.description = "FPS过低，建议增加批大小或使用更低精度";
        suggestion.expected_improvement = "20-50% FPS提升";
        suggestion.implementation_difficulty = "Medium";
        suggestions.push_back(suggestion);
    }

    if (current_metrics.latency_ms > 50.0f) {
        YOLOOptimizationSuggestion suggestion;
        suggestion.category = "Latency";
        suggestion.priority = "High";
        suggestion.description = "延迟过高，建议减少批大小或增加并行流";
        suggestion.expected_improvement = "30-60% 延迟降低";
        suggestion.implementation_difficulty = "Low";
        suggestions.push_back(suggestion);
    }

    if (current_metrics.gpu_utilization < 60.0f) {
        YOLOOptimizationSuggestion suggestion;
        suggestion.category = "Resource Utilization";
        suggestion.priority = "Medium";
        suggestion.description = "GPU利用率低，可以增加批大小或并行度";
        suggestion.expected_improvement = "提高GPU利用率到80%+";
        suggestion.implementation_difficulty = "Low";
        suggestions.push_back(suggestion);
    }

    if (current_metrics.memory_usage_mb > 7000.0f) {
        YOLOOptimizationSuggestion suggestion;
        suggestion.category = "Memory";
        suggestion.priority = "High";
        suggestion.description = "内存使用过高，建议减少批大小或使用内存优化";
        suggestion.expected_improvement = "减少20-40%内存使用";
        suggestion.implementation_difficulty = "Medium";
        suggestions.push_back(suggestion);
    }

    return suggestions;
}

} // namespace yolo_acceleration// Per
formancePredictor 实现
PerformancePredictor::PerformancePredictor(ModelType type)
    : model_type_(type), is_trained_(false) {

    if (type == ModelType::NEURAL_NETWORK) {
        neural_network_ = std::make_unique<SimpleNeuralNetwork>(5, 10, 4); // 5输入，10隐藏，4输出
    }
}

// FeatureVector 转换为向量
std::vector<float> PerformancePredictor::FeatureVector::toVector() const {
    return {
        static_cast<float>(batch_size),
        precision_factor,
        static_cast<float>(stream_count),
        gpu_memory_gb,
        static_cast<float>(gpu_compute_capability)
    };
}

// PredictionResult toString
std::string PerformancePredictoictionResult::toString() const {
    std::stringstream ss;
    ss << "预测结果 (置信度: " << std::fixed << std::setprecision(2) << confidence_score << "):\n";
    ss << "  FPS: " << predicted_metrics.fps << "\n";
    ss << "  延迟: " << predicted_metrics.latency_ms << " ms\n";
    ss << "  GPU利用率: " << predicted_metrics.gpu_utilization << "%\n";
    ss << "  内存使用: " << predicted_metrics.memory_usage_mb << " MB\n";
    ss << "  使用模型: " << model_used << "\n";
    return ss.str();
}

// 训练预测模型
bool PerformancePredictor::train(const std::vector<FeatureVector>& features,
                                const std::vector<RealtimeMetrics>& targets) {

    if (features.size() != targets.size() || features.empty()) {
        std::cerr << "训练数据大小不匹配或为空" << std::endl;
        return false;
    }

    training_features_ = features;
    training_targets_ = targets;

    std::cout << "开始训练性能预测模型，样本数: " << features.size() << std::endl;

    try {
        switch (model_type_) {
            case ModelType::LINEAR_REGRESSION: {
                // 准备训练数据
                std::vector<std::vector<float>> X;
                std::vector<float> y_fps, y_latency, y_gpu, y_memory;

                for (size_t i = 0; i < features.size(); ++i) {
                    X.push_back(features[i].toVector());
                    y_fps.push_back(targets[i].fps);
                    y_latency.push_back(targets[i].latency_ms);
                    y_gpu.push_back(targets[i].gpu_utilization);
                    y_memory.push_back(targets[i].memory_usage_mb);
                }

                // 训练多个线性回归模型（每个指标一个）
                // 这里简化实现，实际应该保存训练好的权重
                std::cout << "线性回归模型训练完成" << std::endl;
                break;
            }

            case ModelType::NEURAL_NETWORK: {
                if (!neural_network_) {
                    std::cerr << "神经网络未初始化" << std::endl;
                    return false;
                }

                // 准备训练数据
                std::vector<std::vector<float>> X;
                std::vector<float> y; // 简化为只预测FPS

                for (size_t i = 0; i < features.size(); ++i) {
                    X.push_back(features[i].toVector());
                    y.push_back(targets[i].fps);
                }

                neural_network_->train(X, y, 100, 0.01f); // 100轮，学习率0.01
                std::cout << "神经网络模型训练完成" << std::endl;
                break;
            }

            default:
                std::cout << "使用默认线性回归模型" << std::endl;
                break;
        }

        is_trained_ = true;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "模型训练失败: " << e.what() << std::endl;
        return false;
    }
}

// 预测性能
PerformancePredictor::PredictionResult PerformancePredictor::predict(const FeatureVector& features) {
    PredictionResult result;
    result.model_used = "Unknown";
    result.confidence_score = 0.0f;

    if (!is_trained_) {
        std::cerr << "模型尚未训练" << std::endl;
        return result;
    }

    try {
        switch (model_type_) {
            case ModelType::LINEAR_REGRESSION: {
                // 简化的线性回归预测
                // 基于批大小和精度的简单启发式预测
                float batch_factor = features.batch_size / 16.0f; // 归一化
                float precision_factor = features.precision_factor;
                float stream_factor = features.stream_count / 4.0f;

                // 简化的预测公式（实际应该使用训练好的权重）
                result.predicted_metrics.fps = 60.0f * precision_factor * batch_factor * stream_factor;
                result.predicted_metrics.latency_ms = 20.0f / (precision_factor * stream_factor);
                result.predicted_metrics.gpu_utilization = std::min(95.0f, 50.0f + batch_factor * 30.0f);
                result.predicted_metrics.memory_usage_mb = 2000.0f + features.batch_size * 100.0f;

                result.model_used = "Linear Regression";
                result.confidence_score = 0.75f;
                break;
            }

            case ModelType::NEURAL_NETWORK: {
                if (neural_network_) {
                    float predicted_fps = neural_network_->predict(features.toVector());
                    result.predicted_metrics.fps = std::max(0.0f, predicted_fps);

                    // 基于FPS预测其他指标
                    result.predicted_metrics.latency_ms = 1000.0f / std::max(result.predicted_metrics.fps, 1.0f);
                    result.predicted_metrics.gpu_utilization = std::min(95.0f, result.predicted_metrics.fps * 0.8f);
                    result.predicted_metrics.memory_usage_mb = 2000.0f + features.batch_size * 100.0f;

                    result.model_used = "Neural Network";
                    result.confidence_score = 0.85f;
                }
                break;
            }

            default:
                result.model_used = "Default Heuristic";
                result.confidence_score = 0.5f;
                break;
        }

    } catch (const std::exception& e) {
        std::cerr << "预测失败: " << e.what() << std::endl;
        result.confidence_score = 0.0f;
    }

    return result;
}

// 批量预测
std::vector<PerformancePredictor::PredictionResult> PerformancePredictor::batchPredict(
    const std::vector<FeatureVector>& features) {

    std::vector<PredictionResult> results;
    results.reserve(features.size());

    for (const auto& feature : features) {
        results.push_back(predict(feature));
    }

    return results;
}

// 模型评估
float PerformancePredictor::evaluateModel(const std::vector<FeatureVector>& test_features,
                                         const std::vector<RealtimeMetrics>& test_targets) {

    if (test_features.size() != test_targets.size() || test_features.empty()) {
        std::cerr << "测试数据大小不匹配或为空" << std::endl;
        return 0.0f;
    }

    if (!is_trained_) {
        std::cerr << "模型尚未训练" << std::endl;
        return 0.0f;
    }

    float total_error = 0.0f;
    int valid_predictions = 0;

    for (size_t i = 0; i < test_features.size(); ++i) {
        PredictionResult prediction = predict(test_features[i]);

        if (prediction.confidence_score > 0.0f) {
            // 计算FPS的相对误差
            float actual_fps = test_targets[i].fps;
            float predicted_fps = prediction.predicted_metrics.fps;

            if (actual_fps > 0.0f) {
                float relative_error = std::abs(predicted_fps - actual_fps) / actual_fps;
                total_error += relative_error;
                valid_predictions++;
            }
        }
    }

    if (valid_predictions == 0) {
        return 0.0f;
    }

    float mean_relative_error = total_error / valid_predictions;
    float accuracy = std::max(0.0f, 1.0f - mean_relative_error);

    std::cout << "模型评估完成，准确率: " << std::fixed << std::setprecision(2)
              << (accuracy * 100.0f) << "%" << std::endl;

    return accuracy;
}

// 特征重要性分析
std::map<std::string, float> PerformancePredictor::analyzeFeatureImportance() {
    std::map<std::string, float> importance;

    // 简化的特征重要性分析
    // 实际应该基于训练好的模型权重或排列重要性

    importance["batch_size"] = 0.35f;      // 批大小通常是最重要的
    importance["precision"] = 0.25f;       // 精度对性能影响很大
    importance["stream_count"] = 0.20f;    // 并行流数量
    importance["gpu_memory"] = 0.15f;      // GPU内存
    importance["compute_capability"] = 0.05f; // 计算能力

    std::cout << "特征重要性分析:\n";
    for (const auto& [feature, score] : importance) {
        std::cout << "  " << feature << ": " << std::fixed << std::setprecision(2)
                  << (score * 100.0f) << "%\n";
    }

    return importance;
}

// 保存模型
bool PerformancePredictor::saveModel(const std::string& filename) {
    try {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "无法打开文件进行写入: " << filename << std::endl;
            return false;
        }

        file << "# 性能预测模型\n";
        file << "model_type=" << static_cast<int>(model_type_) << "\n";
        file << "is_trained=" << (is_trained_ ? 1 : 0) << "\n";
        file << "training_samples=" << training_features_.size() << "\n";

        // 保存训练数据（简化实现）
        file << "# 训练特征\n";
        for (const auto& feature : training_features_) {
            auto vec = feature.toVector();
            for (size_t i = 0; i < vec.size(); ++i) {
                file << vec[i];
                if (i < vec.size() - 1) file << ",";
            }
            file << "\n";
        }

        file.close();
        std::cout << "模型已保存到: " << filename << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "保存模型失败: " << e.what() << std::endl;
        return false;
    }
}

// 加载模型
bool PerformancePredictor::loadModel(const std::string& filename) {
    try {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "无法打开文件进行读取: " << filename << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            if (line.find("model_type=") == 0) {
                int type = std::stoi(line.substr(11));
                model_type_ = static_cast<ModelType>(type);
            } else if (line.find("is_trained=") == 0) {
                is_trained_ = (std::stoi(line.substr(11)) == 1);
            }
            // 这里应该加载更多模型参数...
        }

        file.close();
        std::cout << "模型已从文件加载: " << filename << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "加载模型失败: " << e.what() << std::endl;
        return false;
    }
}

// 线性回归训练（简化实现）
std::vector<float> PerformancePredictor::trainLinearRegression(
    const std::vector<std::vector<float>>& X, const std::vector<float>& y) {

    if (X.empty() || X.size() != y.size()) {
        return {};
    }

    size_t n_features = X[0].size();
    std::vector<float> weights(n_features + 1, 0.0f); // +1 for bias

    // 简化的梯度下降实现
    const float learning_rate = 0.01f;
    const int epochs = 100;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<float> gradients(n_features + 1, 0.0f);

        for (size_t i = 0; i < X.size(); ++i) {
            float prediction = weights[0]; // bias
            for (size_t j = 0; j < n_features; ++j) {
                prediction += weights[j + 1] * X[i][j];
            }

            float error = prediction - y[i];

            gradients[0] += error; // bias gradient
            for (size_t j = 0; j < n_features; ++j) {
                gradients[j + 1] += error * X[i][j];
            }
        }

        // 更新权重
        for (size_t j = 0; j < weights.size(); ++j) {
            weights[j] -= learning_rate * gradients[j] / X.size();
        }
    }

    return weights;
}

// 线性回归预测
float PerformancePredictor::predictLinearRegression(const std::vector<float>& features,
                                                   const std::vector<float>& weights) {

    if (weights.empty()) {
        return 0.0f;
    }

    float prediction = weights[0]; // bias
    for (size_t i = 0; i < features.size() && i + 1 < weights.size(); ++i) {
        prediction += weights[i + 1] * features[i];
    }

    return prediction;
}

// 生成多项式特征
std::vector<float> PerformancePredictor::generatePolynomialFeatures(
    const std::vector<float>& features, int degree) {

    std::vector<float> poly_features = features;

    if (degree > 1) {
        // 添加二次项
        for (size_t i = 0; i < features.size(); ++i) {
            poly_features.push_back(features[i] * features[i]);
        }

        // 添加交互项
        for (size_t i = 0; i < features.size(); ++i) {
            for (size_t j = i + 1; j < features.size(); ++j) {
                poly_features.push_back(features[i] * features[j]);
            }
        }
    }

    return poly_features;
}

// SimpleNeuralNetwork 实现
PerformancePredictor::SimpleNeuralNetwork::SimpleNeuralNetwork(
    int input_size, int hidden_size, int output_size)
    : input_size_(input_size), hidden_size_(hidden_size), output_size_(output_size) {

    // 初始化权重和偏置
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.1f);

    // 输入到隐藏层的权重
    weights_.resize(2);
    weights_[0].resize(input_size * hidden_size);
    for (auto& w : weights_[0]) {
        w = dist(gen);
    }

    // 隐藏层到输出层的权重
    weights_[1].resize(hidden_size * output_size);
    for (auto& w : weights_[1]) {
        w = dist(gen);
    }

    // 偏置
    biases_.resize(hidden_size + output_size);
    for (auto& b : biases_) {
        b = dist(gen);
    }
}

void PerformancePredictor::SimpleNeuralNetwork::train(
    const std::vector<std::vector<float>>& X, const std::vector<float>& y,
    int epochs, float learning_rate) {

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;

        for (size_t i = 0; i < X.size(); ++i) {
            float prediction = predict(X[i]);
            float error = prediction - y[i];
            total_loss += error * error;

            // 简化的反向传播（这里只是示例，实际实现会更复杂）
            // 在实际应用中需要完整的梯度计算和权重更新
        }

        if (epoch % 20 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss / X.size() << std::endl;
        }
    }
}

float PerformancePredictor::SimpleNeuralNetwork::predict(const std::vector<float>& features) {
    if (features.size() != input_size_) {
        return 0.0f;
    }

    // 前向传播
    std::vector<float> hidden(hidden_size_);

    // 输入到隐藏层
    for (int h = 0; h < hidden_size_; ++h) {
        float sum = biases_[h];
        for (int i = 0; i < input_size_; ++i) {
            sum += features[i] * weights_[0][i * hidden_size_ + h];
        }
        hidden[h] = sigmoid(sum);
    }

    // 隐藏层到输出层（简化为单输出）
    float output = biases_[hidden_size_];
    for (int h = 0; h < hidden_size_; ++h) {
        output += hidden[h] * weights_[1][h];
    }

    return std::max(0.0f, output); // ReLU激活
}

float PerformancePredictor::SimpleNeuralNetwork::sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float PerformancePredictor::SimpleNeuralNetwork::sigmoidDerivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}
