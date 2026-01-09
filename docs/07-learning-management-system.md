# 学习管理和进度跟踪系统

## 概述

本文档介绍如何构建一个完整的CUDA学习管理系统，包括学习路径管理、交互式教学界面、进度跟踪和个性化推荐功能。

## 学习目标

完成本模块后，你将能够：
- 设计和实现学习进度跟踪系统
- 开发交互式的CUDA教学界面
- 构建个性化学习路径推荐引擎
- 实现学习成果可视化系统
- 创建完整的学习管理平台

## 学习路径管理系统

### 学习进度跟踪数据库

```cpp
// learning_database.h
#pragma once
#include <sqlite3.h>
#include <string>
#include <vector>
#include <map>
#include <chrono>

struct LearningProgress {
    int user_id;
    std::string module_name;
    std::string lesson_name;
    float completion_percentage;
    int attempts;
    float best_score;
    std::chrono::system_clock::time_point last_accessed;
    std::chrono::duration<double> total_time_spent;
    std::vector<std::string> completed_exercises;
    std::map<std::string, float> skill_scores;
};

struct UserProfile {
    int user_id;
    std::string username;
    std::string email;
    std::string experience_level;  // beginner, intermediate, advanced
    std::vector<std::string> learning_goals;
    std::map<std::string, float> skill_assessments;
    std::chrono::system_clock::time_point registration_date;
    std::chrono::system_clock::time_point last_login;
};

class LearningDatabase {
private:
    sqlite3* db_;
    std::string db_path_;

public:
    LearningDatabase(const std::string& db_path);
    ~LearningDatabase();

    // 初始化数据库
    bool initialize();

    // 用户管理
    bool createUser(const UserProfile& profile);
    bool updateUser(const UserProfile& profile);
    UserProfile getUserProfile(int user_id);
    std::vector<UserProfile> getAllUsers();

    // 学习进度管理
    bool updateProgress(const LearningProgress& progress);
    LearningProgress getProgress(int user_id, const std::string& module_name);
    std::vector<LearningProgress> getUserProgress(int user_id);

    // 技能评估
    bool updateSkillScore(int user_id, const std::string& skill, float score);
    std::map<std::string, float> getSkillScores(int user_id);

    // 学习统计
    std::map<std::string, int> getModuleCompletionStats();
    std::vector<std::pair<std::string, float>> getAverageScores();

private:
    bool createTables();
    bool executeSQL(const std::string& sql);
};

// learning_database.cpp
LearningDatabase::LearningDatabase(const std::string& db_path)
    : db_(nullptr), db_path_(db_path) {}

bool LearningDatabase::initialize() {
    int rc = sqlite3_open(db_path_.c_str(), &db_);
    if (rc != SQLITE_OK) {
        return false;
    }

    return createTables();
}

bool LearningDatabase::createTables() {
    std::vector<std::string> create_statements = {
        R"(
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            experience_level TEXT NOT NULL,
            learning_goals TEXT,
            registration_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_login DATETIME
        )
        )",

        R"(
        CREATE TABLE IF NOT EXISTS learning_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            module_name TEXT NOT NULL,
            lesson_name TEXT NOT NULL,
            completion_percentage REAL DEFAULT 0.0,
            attempts INTEGER DEFAULT 0,
            best_score REAL DEFAULT 0.0,
            last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
            total_time_spent REAL DEFAULT 0.0,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        )",

        R"(
        CREATE TABLE IF NOT EXISTS skill_assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            skill_name TEXT NOT NULL,
            score REAL NOT NULL,
            assessment_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        )",

        R"(
        CREATE TABLE IF NOT EXISTS exercise_completions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            exercise_name TEXT NOT NULL,
            completion_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            score REAL,
            time_taken REAL,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        )"
    };

    for (const auto& sql : create_statements) {
        if (!executeSQL(sql)) {
            return false;
        }
    }

    return true;
}
```

### 技能评估和推荐系统

```cpp
// skill_assessment.h
#pragma once
#include "learning_database.h"
#include <vector>
#include <string>
#include <map>

struct SkillArea {
    std::string name;
    std::string description;
    std::vector<std::string> prerequisites;
    std::vector<std::string> learning_objectives;
    float difficulty_level;  // 1.0 - 5.0
};

struct AssessmentQuestion {
    std::string question_id;
    std::string skill_area;
    std::string question_text;
    std::vector<std::string> options;
    std::string correct_answer;
    std::string explanation;
    float difficulty;
    std::string code_snippet;  // 可选的代码示例
};

struct LearningRecommendation {
    std::string module_name;
    std::string reason;
    float priority_score;
    std::vector<std::string> prerequisites_needed;
    std::chrono::duration<double> estimated_time;
};

class SkillAssessmentEngine {
private:
    LearningDatabase* db_;
    std::vector<SkillArea> skill_areas_;
    std::vector<AssessmentQuestion> question_bank_;

public:
    SkillAssessmentEngine(LearningDatabase* db);

    // 初始化技能领域和题库
    void initializeSkillAreas();
    void loadQuestionBank();

    // 技能评估
    std::vector<AssessmentQuestion> generateAssessment(int user_id,
                                                      const std::string& skill_area);
    float calculateSkillScore(const std::vector<std::pair<std::string, std::string>>& answers);
    void updateUserSkillProfile(int user_id, const std::map<std::string, float>& scores);

    // 学习路径推荐
    std::vector<LearningRecommendation> generateRecommendations(int user_id);
    std::vector<std::string> getOptimalLearningPath(int user_id);

    // 个性化内容推荐
    std::vector<std::string> recommendExercises(int user_id, const std::string& skill_area);
    std::vector<std::string> recommendProjects(int user_id);

private:
    float assessSkillLevel(int user_id, const std::string& skill_area);
    std::vector<std::string> identifyWeakAreas(int user_id);
    float calculatePrerequisiteScore(int user_id, const std::vector<std::string>& prerequisites);
};

// skill_assessment.cpp
void SkillAssessmentEngine::initializeSkillAreas() {
    skill_areas_ = {
        {
            "cuda_basics",
            "CUDA编程基础知识",
            {},
            {"理解GPU架构", "掌握CUDA编程模型", "编写简单核函数"},
            1.0f
        },
        {
            "memory_management",
            "GPU内存管理",
            {"cuda_basics"},
            {"掌握不同内存类型", "优化内存访问模式", "避免内存泄漏"},
            2.5f
        },
        {
            "kernel_optimization",
            "核函数优化",
            {"cuda_basics", "memory_management"},
            {"优化线程配置", "提高占用率", "减少分支发散"},
            3.0f
        },
        {
            "advanced_techniques",
            "高级优化技术",
            {"kernel_optimization"},
            {"使用共享内存", "warp级优化", "多GPU编程"},
            4.0f
        },
        {
            "deep_learning_acceleration",
            "深度学习加速",
            {"kernel_optimization", "advanced_techniques"},
            {"实现自定义算子", "优化神经网络推理", "集成深度学习框架"},
            4.5f
        }
    };
}

std::vector<LearningRecommendation> SkillAssessmentEngine::generateRecommendations(int user_id) {
    std::vector<LearningRecommendation> recommendations;

    // 获取用户当前技能水平
    auto skill_scores = db_->getSkillScores(user_id);
    auto user_profile = db_->getUserProfile(user_id);

    for (const auto& skill_area : skill_areas_) {
        float current_score = skill_scores[skill_area.name];
        float prerequisite_score = calculatePrerequisiteScore(user_id, skill_area.prerequisites);

        // 如果前置技能满足且当前技能有提升空间
        if (prerequisite_score >= 0.7f && current_score < 0.8f) {
            LearningRecommendation rec;
            rec.module_name = skill_area.name;
            rec.priority_score = (0.8f - current_score) * prerequisite_score;

            // 根据经验水平调整推荐
            if (user_profile.experience_level == "beginner" && skill_area.difficulty_level > 2.0f) {
                rec.priority_score *= 0.5f;  // 降低难度过高内容的优先级
            }

            rec.reason = "基于您的当前技能水平，建议学习此模块";
            rec.estimated_time = std::chrono::hours(static_cast<int>(skill_area.difficulty_level * 2));

            recommendations.push_back(rec);
        }
    }

    // 按优先级排序
    std::sort(recommendations.begin(), recommendations.end(),
              [](const LearningRecommendation& a, const LearningRecommendation& b) {
                  return a.priority_score > b.priority_score;
              });

    return recommendations;
}
```

### 个性化学习路径生成器

```cpp
// learning_path_generator.h
#pragma once
#include "skill_assessment.h"
#include <graph>
#include <algorithm>

struct LearningModule {
    std::string module_id;
    std::string name;
    std::string description;
    std::vector<std::string> prerequisites;
    std::vector<std::string> learning_outcomes;
    std::chrono::duration<double> estimated_duration;
    float difficulty_level;
    std::vector<std::string> exercises;
    std::vector<std::string> projects;
};

struct LearningPath {
    std::string path_id;
    std::string name;
    std::vector<std::string> module_sequence;
    std::chrono::duration<double> total_duration;
    std::string target_audience;
    std::vector<std::string> learning_goals;
};

class LearningPathGenerator {
private:
    std::vector<LearningModule> modules_;
    std::vector<LearningPath> predefined_paths_;
    SkillAssessmentEngine* assessment_engine_;

public:
    LearningPathGenerator(SkillAssessmentEngine* engine);

    // 初始化学习模块
    void initializeModules();
    void initializePredefinedPaths();

    // 个性化路径生成
    LearningPath generatePersonalizedPath(int user_id);
    LearningPath adaptPathToUserLevel(const LearningPath& base_path, int user_id);

    // 路径优化
    std::vector<std::string> optimizeModuleSequence(const std::vector<std::string>& modules,
                                                   int user_id);

    // 动态路径调整
    LearningPath adjustPathBasedOnProgress(int user_id, const std::string& current_path_id);

    // 路径推荐
    std::vector<LearningPath> recommendPaths(int user_id);

private:
    std::vector<std::string> topologicalSort(const std::vector<std::string>& modules);
    float calculatePathDifficulty(const std::vector<std::string>& modules);
    bool checkPrerequisites(int user_id, const std::string& module_id);
};

// learning_path_generator.cpp
void LearningPathGenerator::initializeModules() {
    modules_ = {
        {
            "cuda_intro",
            "CUDA编程入门",
            "学习CUDA编程的基础概念和开发环境搭建",
            {},
            {"理解GPU架构", "搭建CUDA开发环境", "编写第一个CUDA程序"},
            std::chrono::hours(8),
            1.0f,
            {"hello_world", "vector_addition", "basic_kernels"},
            {"simple_image_processing"}
        },
        {
            "memory_basics",
            "内存管理基础",
            "学习CUDA内存模型和基本内存管理技术",
            {"cuda_intro"},
            {"掌握全局内存使用", "理解内存合并", "使用共享内存"},
            std::chrono::hours(12),
            2.0f,
            {"memory_allocation", "coalesced_access", "shared_memory_basics"},
            {"matrix_multiplication"}
        },
        {
            "kernel_optimization",
            "核函数优化",
            "学习核函数性能优化技术",
            {"memory_basics"},
            {"优化线程配置", "提高占用率", "减少分支发散"},
            std::chrono::hours(16),
            3.0f,
            {"occupancy_optimization", "branch_optimization", "warp_efficiency"},
            {"optimized_convolution"}
        },
        {
            "advanced_memory",
            "高级内存技术",
            "学习纹理内存、常量内存等高级内存技术",
            {"kernel_optimization"},
            {"使用纹理内存", "优化常量内存访问", "统一内存管理"},
            std::chrono::hours(14),
            3.5f,
            {"texture_memory", "constant_memory", "unified_memory"},
            {"advanced_image_processing"}
        },
        {
            "multi_gpu",
            "多GPU编程",
            "学习多GPU协同编程技术",
            {"advanced_memory"},
            {"NCCL通信", "负载均衡", "分布式计算"},
            std::chrono::hours(20),
            4.0f,
            {"nccl_basics", "multi_gpu_reduction", "distributed_training"},
            {"multi_gpu_deep_learning"}
        }
    };
}

LearningPath LearningPathGenerator::generatePersonalizedPath(int user_id) {
    auto user_profile = assessment_engine_->db_->getUserProfile(user_id);
    auto skill_scores = assessment_engine_->db_->getSkillScores(user_id);

    LearningPath personalized_path;
    personalized_path.path_id = "personalized_" + std::to_string(user_id);
    personalized_path.name = "个性化学习路径";
    personalized_path.target_audience = user_profile.experience_level;

    // 根据用户技能水平选择起始模块
    std::vector<std::string> selected_modules;

    for (const auto& module : modules_) {
        bool should_include = false;

        // 检查前置条件
        if (checkPrerequisites(user_id, module.module_id)) {
            // 根据技能分数决定是否包含
            float relevant_score = 0.0f;
            for (const auto& outcome : module.learning_outcomes) {
                // 简化：假设学习成果对应技能名称
                if (skill_scores.find(outcome) != skill_scores.end()) {
                    relevant_score = std::max(relevant_score, skill_scores[outcome]);
                }
            }

            // 如果技能分数低于阈值，包含此模块
            if (relevant_score < 0.7f) {
                should_include = true;
            }
        }

        if (should_include) {
            selected_modules.push_back(module.module_id);
        }
    }

    // 优化模块顺序
    personalized_path.module_sequence = optimizeModuleSequence(selected_modules, user_id);

    // 计算总时长
    personalized_path.total_duration = std::chrono::hours(0);
    for (const auto& module_id : personalized_path.module_sequence) {
        auto it = std::find_if(modules_.begin(), modules_.end(),
                              [&module_id](const LearningModule& m) {
                                  return m.module_id == module_id;
                              });
        if (it != modules_.end()) {
            personalized_path.total_duration += it->estimated_duration;
        }
    }

    return personalized_path;
}

std::vector<std::string> LearningPathGenerator::optimizeModuleSequence(
    const std::vector<std::string>& modules, int user_id) {

    // 构建依赖图
    std::map<std::string, std::vector<std::string>> dependencies;
    std::map<std::string, int> in_degree;

    for (const auto& module_id : modules) {
        auto it = std::find_if(modules_.begin(), modules_.end(),
                              [&module_id](const LearningModule& m) {
                                  return m.module_id == module_id;
                              });

        if (it != modules_.end()) {
            in_degree[module_id] = 0;
            for (const auto& prereq : it->prerequisites) {
                if (std::find(modules.begin(), modules.end(), prereq) != modules.end()) {
                    dependencies[prereq].push_back(module_id);
                    in_degree[module_id]++;
                }
            }
        }
    }

    // 拓扑排序
    std::queue<std::string> queue;
    std::vector<std::string> result;

    for (const auto& pair : in_degree) {
        if (pair.second == 0) {
            queue.push(pair.first);
        }
    }

    while (!queue.empty()) {
        std::string current = queue.front();
        queue.pop();
        result.push_back(current);

        for (const auto& dependent : dependencies[current]) {
            in_degree[dependent]--;
            if (in_degree[dependent] == 0) {
                queue.push(dependent);
            }
        }
    }

    return result;
}
```

## 交互式教学界面

### Web界面框架

```html
<!-- learning_interface.html -->
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CUDA学习管理系统</title>
    <link rel="stylesheet" href="styles.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.34.1/min/vs/loader.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div class="container">
        <!-- 导航栏 -->
        <nav class="navbar">
            <div class="nav-brand">CUDA学习系统</div>
            <ul class="nav-menu">
                <li><a href="#dashboard">仪表板</a></li>
                <li><a href="#learning">学习模块</a></li>
                <li><a href="#exercises">练习</a></li>
                <li><a href="#projects">项目</a></li>
                <li><a href="#progress">进度</a></li>
            </ul>
        </nav>

        <!-- 主要内容区域 -->
        <main class="main-content">
            <!-- 仪表板 -->
            <section id="dashboard" class="content-section">
                <h2>学习仪表板</h2>
                <div class="dashboard-grid">
                    <div class="stat-card">
                        <h3>总体进度</h3>
                        <div class="progress-circle" id="overall-progress"></div>
                    </div>
                    <div class="stat-card">
                        <h3>当前模块</h3>
                        <p id="current-module">内存管理基础</p>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: 65%"></div>
                        </div>
                    </div>
                    <div class="stat-card">
                        <h3>技能雷达图</h3>
                        <div id="skill-radar"></div>
                    </div>
                    <div class="stat-card">
                        <h3>学习建议</h3>
                        <ul id="recommendations"></ul>
                    </div>
                </div>
            </section>

            <!-- 学习模块 -->
            <section id="learning" class="content-section">
                <h2>学习模块</h2>
                <div class="module-list">
                    <div class="module-card completed">
                        <h3>CUDA编程入门</h3>
                        <p>学习CUDA编程的基础概念</p>
                        <div class="module-progress">
                            <span>已完成</span>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: 100%"></div>
                            </div>
                        </div>
                    </div>
                    <div class="module-card current">
                        <h3>内存管理基础</h3>
                        <p>学习CUDA内存模型和管理技术</p>
                        <div class="module-progress">
                            <span>进行中 (65%)</span>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: 65%"></div>
                            </div>
                        </div>
                        <button class="continue-btn">继续学习</button>
                    </div>
                    <div class="module-card locked">
                        <h3>核函数优化</h3>
                        <p>学习核函数性能优化技术</p>
                        <div class="module-progress">
                            <span>未解锁</span>
                        </div>
                    </div>
                </div>
            </section>

            <!-- 代码编辑器和练习 -->
            <section id="exercises" class="content-section">
                <h2>交互式练习</h2>
                <div class="exercise-container">
                    <div class="exercise-description">
                        <h3>练习：向量加法优化</h3>
                        <p>实现一个优化的向量加法核函数，要求：</p>
                        <ul>
                            <li>使用合并内存访问</li>
                            <li>处理任意大小的向量</li>
                            <li>添加错误检查</li>
                        </ul>
                    </div>
                    <div class="code-editor-container">
                        <div id="code-editor"></div>
                        <div class="editor-controls">
                            <button id="run-code">运行代码</button>
                            <button id="submit-code">提交答案</button>
                            <button id="get-hint">获取提示</button>
                        </div>
                    </div>
                    <div class="output-container">
                        <h4>输出结果</h4>
                        <pre id="code-output"></pre>
                    </div>
                </div>
            </section>

            <!-- 性能可视化 -->
            <section id="performance" class="content-section">
                <h2>性能分析</h2>
                <div class="performance-dashboard">
                    <div class="chart-container">
                        <h3>执行时间对比</h3>
                        <div id="performance-chart"></div>
                    </div>
                    <div class="metrics-container">
                        <h3>性能指标</h3>
                        <div class="metric-item">
                            <span>内存带宽利用率</span>
                            <div class="metric-bar">
                                <div class="metric-fill" style="width: 78%">78%</div>
                            </div>
                        </div>
                        <div class="metric-item">
                            <span>GPU占用率</span>
                            <div class="metric-bar">
                                <div class="metric-fill" style="width: 85%">85%</div>
                            </div>
                        </div>
                        <div class="metric-item">
                            <span>计算效率</span>
                            <div class="metric-bar">
                                <div class="metric-fill" style="width: 92%">92%</div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>
        </main>
    </div>

    <script src="learning_interface.js"></script>
</body>
</html>
```

### JavaScript交互逻辑

```javascript
// learning_interface.js
class LearningInterface {
    constructor() {
        this.currentUser = null;
        this.currentModule = null;
        this.codeEditor = null;
        this.initializeInterface();
    }

    async initializeInterface() {
        // 初始化代码编辑器
        this.initializeCodeEditor();

        // 加载用户数据
        await this.loadUserData();

        // 初始化图表
        this.initializeCharts();

        // 绑定事件监听器
        this.bindEventListeners();

        // 加载学习内容
        this.loadLearningContent();
    }

    initializeCodeEditor() {
        require.config({ paths: { vs: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.34.1/min/vs' }});
        require(['vs/editor/editor.main'], () => {
            this.codeEditor = monaco.editor.create(document.getElementById('code-editor'), {
                value: `// CUDA向量加法示例
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int n = 1000000;
    size_t size = n * sizeof(float);

    // 在这里实现你的代码

    return 0;
}`,
                language: 'cpp',
                theme: 'vs-dark',
                automaticLayout: true,
                fontSize: 14,
                minimap: { enabled: false }
            });
        });
    }

    async loadUserData() {
        try {
            const response = await fetch('/api/user/profile');
            this.currentUser = await response.json();
            this.updateDashboard();
        } catch (error) {
            console.error('Failed to load user data:', error);
        }
    }

    updateDashboard() {
        // 更新总体进度
        this.updateOverallProgress(this.currentUser.overall_progress);

        // 更新当前模块
        document.getElementById('current-module').textContent = this.currentUser.current_module;

        // 更新技能雷达图
        this.updateSkillRadar(this.currentUser.skill_scores);

        // 更新学习建议
        this.updateRecommendations(this.currentUser.recommendations);
    }

    updateOverallProgress(progress) {
        const progressElement = document.getElementById('overall-progress');
        const circumference = 2 * Math.PI * 45; // 假设半径为45
        const offset = circumference - (progress / 100) * circumference;

        progressElement.innerHTML = `
            <svg width="120" height="120">
                <circle cx="60" cy="60" r="45" stroke="#e0e0e0" stroke-width="8" fill="none"/>
                <circle cx="60" cy="60" r="45" stroke="#4CAF50" stroke-width="8" fill="none"
                        stroke-dasharray="${circumference}" stroke-dashoffset="${offset}"
                        transform="rotate(-90 60 60)"/>
                <text x="60" y="65" text-anchor="middle" font-size="18" font-weight="bold">${progress}%</text>
            </svg>
        `;
    }

    updateSkillRadar(skillScores) {
        const data = [{
            type: 'scatterpolar',
            r: Object.values(skillScores).map(score => score * 100),
            theta: Object.keys(skillScores),
            fill: 'toself',
            name: '技能水平'
        }];

        const layout = {
            polar: {
                radialaxis: {
                    visible: true,
                    range: [0, 100]
                }
            },
            showlegend: false,
            width: 300,
            height: 300
        };

        Plotly.newPlot('skill-radar', data, layout);
    }

    updateRecommendations(recommendations) {
        const recommendationsList = document.getElementById('recommendations');
        recommendationsList.innerHTML = '';

        recommendations.forEach(rec => {
            const li = document.createElement('li');
            li.innerHTML = `
                <strong>${rec.module_name}</strong>
                <p>${rec.reason}</p>
                <small>预计时间: ${rec.estimated_time}</small>
            `;
            recommendationsList.appendChild(li);
        });
    }

    bindEventListeners() {
        // 运行代码按钮
        document.getElementById('run-code').addEventListener('click', () => {
            this.runCode();
        });

        // 提交代码按钮
        document.getElementById('submit-code').addEventListener('click', () => {
            this.submitCode();
        });

        // 获取提示按钮
        document.getElementById('get-hint').addEventListener('click', () => {
            this.getHint();
        });

        // 导航菜单
        document.querySelectorAll('.nav-menu a').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                this.showSection(e.target.getAttribute('href').substring(1));
            });
        });

        // 继续学习按钮
        document.querySelectorAll('.continue-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.continueModule(e.target.closest('.module-card'));
            });
        });
    }

    async runCode() {
        const code = this.codeEditor.getValue();
        const outputElement = document.getElementById('code-output');

        outputElement.textContent = '正在编译和运行...';

        try {
            const response = await fetch('/api/code/run', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ code: code })
            });

            const result = await response.json();

            if (result.success) {
                outputElement.innerHTML = `
                    <div class="output-success">
                        <h5>编译成功</h5>
                        <pre>${result.output}</pre>
                        <div class="performance-metrics">
                            <p>执行时间: ${result.execution_time}ms</p>
                            <p>内存使用: ${result.memory_usage}MB</p>
                        </div>
                    </div>
                `;

                // 更新性能图表
                this.updatePerformanceChart(result.performance_data);
            } else {
                outputElement.innerHTML = `
                    <div class="output-error">
                        <h5>编译错误</h5>
                        <pre>${result.error}</pre>
                    </div>
                `;
            }
        } catch (error) {
            outputElement.innerHTML = `
                <div class="output-error">
                    <h5>运行失败</h5>
                    <pre>${error.message}</pre>
                </div>
            `;
        }
    }

    async submitCode() {
        const code = this.codeEditor.getValue();

        try {
            const response = await fetch('/api/exercise/submit', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    code: code,
                    exercise_id: this.currentExercise.id
                })
            });

            const result = await response.json();

            if (result.passed) {
                this.showSuccessMessage('恭喜！练习完成！', result.feedback);
                this.updateProgress(result.progress_update);
            } else {
                this.showErrorMessage('练习未通过', result.feedback);
            }
        } catch (error) {
            this.showErrorMessage('提交失败', error.message);
        }
    }

    async getHint() {
        try {
            const response = await fetch(`/api/exercise/hint/${this.currentExercise.id}`);
            const hint = await response.json();

            this.showHintModal(hint.content);
        } catch (error) {
            console.error('Failed to get hint:', error);
        }
    }

    updatePerformanceChart(performanceData) {
        const trace = {
            x: performanceData.labels,
            y: performanceData.values,
            type: 'bar',
            marker: {
                color: 'rgba(76, 175, 80, 0.8)'
            }
        };

        const layout = {
            title: '性能对比',
            xaxis: { title: '实现版本' },
            yaxis: { title: '执行时间 (ms)' }
        };

        Plotly.newPlot('performance-chart', [trace], layout);
    }

    showSection(sectionId) {
        // 隐藏所有内容区域
        document.querySelectorAll('.content-section').forEach(section => {
            section.style.display = 'none';
        });

        // 显示目标区域
        const targetSection = document.getElementById(sectionId);
        if (targetSection) {
            targetSection.style.display = 'block';
        }

        // 更新导航状态
        document.querySelectorAll('.nav-menu a').forEach(link => {
            link.classList.remove('active');
        });
        document.querySelector(`[href="#${sectionId}"]`).classList.add('active');
    }

    showSuccessMessage(title, message) {
        // 实现成功消息显示
        const modal = this.createModal(title, message, 'success');
        document.body.appendChild(modal);
    }

    showErrorMessage(title, message) {
        // 实现错误消息显示
        const modal = this.createModal(title, message, 'error');
        document.body.appendChild(modal);
    }

    createModal(title, content, type) {
        const modal = document.createElement('div');
        modal.className = `modal ${type}`;
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>${title}</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <p>${content}</p>
                </div>
            </div>
        `;

        // 绑定关闭事件
        modal.querySelector('.modal-close').addEventListener('click', () => {
            modal.remove();
        });

        return modal;
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    new LearningInterface();
});
```

### 后端API服务

```cpp
// learning_api_server.h
#pragma once
#include "learning_database.h"
#include "skill_assessment.h"
#include "learning_path_generator.h"
#include <httplib.h>
#include <nlohmann/json.hpp>

class LearningAPIServer {
private:
    std::unique_ptr<httplib::Server> server_;
    std::unique_ptr<LearningDatabase> db_;
    std::unique_ptr<SkillAssessmentEngine> assessment_engine_;
    std::unique_ptr<LearningPathGenerator> path_generator_;

public:
    LearningAPIServer(int port = 8080);
    ~LearningAPIServer();

    // 启动服务器
    void start();
    void stop();

private:
    // 初始化路由
    void setupRoutes();

    // 用户相关API
    void handleGetUserProfile(const httplib::Request& req, httplib::Response& res);
    void handleUpdateUserProfile(const httplib::Request& req, httplib::Response& res);

    // 学习进度API
    void handleGetProgress(const httplib::Request& req, httplib::Response& res);
    void handleUpdateProgress(const httplib::Request& req, httplib::Response& res);

    // 代码执行API
    void handleRunCode(const httplib::Request& req, httplib::Response& res);
    void handleSubmitExercise(const httplib::Request& req, httplib::Response& res);

    // 学习路径API
    void handleGetLearningPath(const httplib::Request& req, httplib::Response& res);
    void handleGeneratePersonalizedPath(const httplib::Request& req, httplib::Response& res);

    // 技能评估API
    void handleGetAssessment(const httplib::Request& req, httplib::Response& res);
    void handleSubmitAssessment(const httplib::Request& req, httplib::Response& res);

    // 工具函数
    nlohmann::json userProfileToJson(const UserProfile& profile);
    nlohmann::json learningProgressToJson(const LearningProgress& progress);
    nlohmann::json learningPathToJson(const LearningPath& path);

    // 代码执行
    struct CodeExecutionResult {
        bool success;
        std::string output;
        std::string error;
        double execution_time;
        size_t memory_usage;
        std::map<std::string, double> performance_metrics;
    };

    CodeExecutionResult executeCode(const std::string& code);
};

// learning_api_server.cpp
void LearningAPIServer::setupRoutes() {
    // 静态文件服务
    server_->set_mount_point("/", "./web");

    // 用户API
    server_->Get("/api/user/profile", [this](const httplib::Request& req, httplib::Response& res) {
        handleGetUserProfile(req, res);
    });

    server_->Post("/api/user/profile", [this](const httplib::Request& req, httplib::Response& res) {
        handleUpdateUserProfile(req, res);
    });

    // 学习进度API
    server_->Get("/api/progress", [this](const httplib::Request& req, httplib::Response& res) {
        handleGetProgress(req, res);
    });

    server_->Post("/api/progress", [this](const httplib::Request& req, httplib::Response& res) {
        handleUpdateProgress(req, res);
    });

    // 代码执行API
    server_->Post("/api/code/run", [this](const httplib::Request& req, httplib::Response& res) {
        handleRunCode(req, res);
    });

    server_->Post("/api/exercise/submit", [this](const httplib::Request& req, httplib::Response& res) {
        handleSubmitExercise(req, res);
    });

    // 学习路径API
    server_->Get("/api/learning-path", [this](const httplib::Request& req, httplib::Response& res) {
        handleGetLearningPath(req, res);
    });

    server_->Post("/api/learning-path/generate", [this](const httplib::Request& req, httplib::Response& res) {
        handleGeneratePersonalizedPath(req, res);
    });
}

void LearningAPIServer::handleRunCode(const httplib::Request& req, httplib::Response& res) {
    try {
        auto json_body = nlohmann::json::parse(req.body);
        std::string code = json_body["code"];

        auto result = executeCode(code);

        nlohmann::json response;
        response["success"] = result.success;
        response["output"] = result.output;
        response["error"] = result.error;
        response["execution_time"] = result.execution_time;
        response["memory_usage"] = result.memory_usage;
        response["performance_data"] = {
            {"labels", {"CPU版本", "GPU版本", "优化版本"}},
            {"values", {100.0, result.execution_time, result.execution_time * 0.8}}
        };

        res.set_content(response.dump(), "application/json");
    } catch (const std::exception& e) {
        nlohmann::json error_response;
        error_response["success"] = false;
        error_response["error"] = e.what();

        res.status = 400;
        res.set_content(error_response.dump(), "application/json");
    }
}

LearningAPIServer::CodeExecutionResult LearningAPIServer::executeCode(const std::string& code) {
    CodeExecutionResult result;

    // 将代码写入临时文件
    std::string temp_file = "/tmp/cuda_code_" + std::to_string(time(nullptr)) + ".cu";
    std::ofstream file(temp_file);
    file << code;
    file.close();

    // 编译代码
    std::string compile_cmd = "nvcc -o /tmp/cuda_program " + temp_file + " 2>&1";
    std::string compile_output = executeCommand(compile_cmd);

    if (compile_output.find("error") != std::string::npos) {
        result.success = false;
        result.error = compile_output;
        return result;
    }

    // 执行程序并测量性能
    auto start_time = std::chrono::high_resolution_clock::now();

    std::string run_cmd = "/tmp/cuda_program 2>&1";
    result.output = executeCommand(run_cmd);

    auto end_time = std::chrono::high_resolution_clock::now();
    result.execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    result.success = true;

    // 清理临时文件
    std::remove(temp_file.c_str());
    std::remove("/tmp/cuda_program");

    return result;
}
```

## 学习成果可视化

### 进度跟踪图表

```javascript
// progress_visualization.js
class ProgressVisualization {
    constructor() {
        this.charts = {};
        this.initializeCharts();
    }

    initializeCharts() {
        this.createProgressTimeline();
        this.createSkillProgressChart();
        this.createPerformanceComparisonChart();
        this.createLearningHeatmap();
    }

    createProgressTimeline() {
        const timelineData = {
            x: ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05'],
            y: [10, 25, 45, 70, 85],
            type: 'scatter',
            mode: 'lines+markers',
            name: '学习进度',
            line: { color: '#4CAF50', width: 3 },
            marker: { size: 8 }
        };

        const layout = {
            title: '学习进度时间线',
            xaxis: { title: '时间' },
            yaxis: { title: '完成百分比 (%)' },
            showlegend: false
        };

        Plotly.newPlot('progress-timeline', [timelineData], layout);
        this.charts['timeline'] = timelineData;
    }

    createSkillProgressChart() {
        const skills = ['CUDA基础', '内存管理', '核函数优化', '性能分析', '多GPU编程'];
        const currentScores = [90, 75, 60, 45, 20];
        const targetScores = [95, 85, 80, 75, 70];

        const currentTrace = {
            x: skills,
            y: currentScores,
            type: 'bar',
            name: '当前水平',
            marker: { color: '#2196F3' }
        };

        const targetTrace = {
            x: skills,
            y: targetScores,
            type: 'bar',
            name: '目标水平',
            marker: { color: '#FFC107', opacity: 0.6 }
        };

        const layout = {
            title: '技能发展对比',
            xaxis: { title: '技能领域' },
            yaxis: { title: '掌握程度 (%)' },
            barmode: 'group'
        };

        Plotly.newPlot('skill-progress', [currentTrace, targetTrace], layout);
    }

    createPerformanceComparisonChart() {
        const exercises = ['向量加法', '矩阵乘法', '卷积运算', '归约操作', '排序算法'];
        const initialTimes = [100, 200, 500, 150, 300];
        const optimizedTimes = [20, 45, 80, 25, 60];

        const initialTrace = {
            x: exercises,
            y: initialTimes,
            type: 'bar',
            name: '初始实现',
            marker: { color: '#F44336' }
        };

        const optimizedTrace = {
            x: exercises,
            y: optimizedTimes,
            type: 'bar',
            name: '优化后',
            marker: { color: '#4CAF50' }
        };

        const layout = {
            title: '性能优化对比',
            xaxis: { title: '练习项目' },
            yaxis: { title: '执行时间 (ms)' },
            barmode: 'group'
        };

        Plotly.newPlot('performance-comparison', [initialTrace, optimizedTrace], layout);
    }

    createLearningHeatmap() {
        const days = ['周一', '周二', '周三', '周四', '周五', '周六', '周日'];
        const weeks = ['第1周', '第2周', '第3周', '第4周'];

        const activityData = [
            [2, 3, 1, 4, 2, 1, 0],
            [3, 4, 2, 3, 4, 2, 1],
            [1, 2, 4, 5, 3, 2, 1],
            [4, 3, 3, 2, 4, 3, 2]
        ];

        const heatmapTrace = {
            z: activityData,
            x: days,
            y: weeks,
            type: 'heatmap',
            colorscale: 'Greens',
            showscale: true
        };

        const layout = {
            title: '学习活跃度热力图',
            xaxis: { title: '星期' },
            yaxis: { title: '周次' }
        };

        Plotly.newPlot('learning-heatmap', [heatmapTrace], layout);
    }

    updateProgressData(newData) {
        // 更新进度数据
        this.charts.timeline.x.push(newData.date);
        this.charts.timeline.y.push(newData.progress);

        Plotly.redraw('progress-timeline');
    }

    generateProgressReport() {
        const report = {
            totalStudyTime: this.calculateTotalStudyTime(),
            completedModules: this.getCompletedModules(),
            skillImprovements: this.calculateSkillImprovements(),
            performanceGains: this.calculatePerformanceGains(),
            recommendations: this.generateRecommendations()
        };

        return report;
    }

    calculateTotalStudyTime() {
        // 计算总学习时间
        return "45小时30分钟";
    }

    getCompletedModules() {
        // 获取已完成模块
        return ["CUDA编程入门", "内存管理基础"];
    }

    calculateSkillImprovements() {
        // 计算技能提升
        return {
            "CUDA基础": "+15%",
            "内存管理": "+20%",
            "核函数优化": "+10%"
        };
    }

    calculatePerformanceGains() {
        // 计算性能提升
        return {
            "平均加速比": "3.2x",
            "最佳加速比": "8.5x",
            "内存效率提升": "45%"
        };
    }

    generateRecommendations() {
        // 生成学习建议
        return [
            "继续深入学习核函数优化技术",
            "加强性能分析工具的使用",
            "开始学习多GPU编程"
        ];
    }
}

// 初始化可视化
const progressViz = new ProgressVisualization();
```

## 实践练习

### 练习1：用户进度跟踪
实现一个完整的用户学习进度跟踪系统，包括数据库设计和API接口。

### 练习2：个性化推荐算法
开发基于协同过滤的学习内容推荐算法。

### 练习3：交互式代码编辑器
集成Monaco Editor，实现CUDA代码的语法高亮和自动补全。

### 练习4：学习分析仪表板
创建一个全面的学习分析仪表板，展示各种学习指标。

## 总结

本模块介绍了完整的CUDA学习管理系统，包括：
- 学习进度跟踪和数据管理
- 个性化学习路径生成
- 交互式教学界面开发
- 学习成果可视化系统
- 技能评估和推荐引擎

通过这个系统，学习者可以获得个性化的学习体验，教师可以跟踪学习效果，从而提高CUDA编程教学的质量和效率。

---

**下一步**：学习[集成部署和文档系统](08-deployment-documentation.md)
