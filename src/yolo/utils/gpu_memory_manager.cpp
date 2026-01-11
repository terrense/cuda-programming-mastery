#include "gpu_memory_manager.h"
#include "../../core/error_handler.h"
#include <iostream>
#include <algorithm>
#include <cstring>

namespace cuda_learning {
namespace yolo {

// GPUMemoryManager 实现
GPUMemoryManager::GPUMemoryManager()
    : total_allocated_(0), peak_usage_(0), current_usage_(0) {
    std::cout << "GPU Memory Manager initialized" << std::endl;
}

GPUMemoryManager::~GPUMemoryManager() {
    cleanup();
    std::cout << "GPU Memory Manager destroyed" << std::endl;
}

void* GPUMemoryManager::allocate(size_t size, const std::string& tag, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(memory_mutex_);

    size_t aligned_size = alignSize(size);

    // 查找合适的空闲块
    MemoryBlock* block = findFreeBlock(aligned_size);

    if (block) {
        block->in_use = true;
        block->stream = stream;
        block->tag = tag;

        // 如果块太大，分割它
        if (block->size > aligned_size * 2) {
            splitBlock(block, aligned_size);
        }

        current_usage_ += block->size;
        peak_usage_ = std::max(peak_usage_, current_usage_);

        std::cout << "Allocated " << aligned_size << " bytes (tag: " << tag << ")" << std::endl;
        return block->ptr;
    }

    // 没有合适的块，分配新内存
    void* ptr;
    cudaError_t error = cudaMalloc(&ptr, aligned_size);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory: " << cudaGetErrorString(error) << std::endl;
        return nullptr;
    }

    MemoryBlock new_block(ptr, aligned_size, tag);
    new_block.in_use = true;
    new_block.stream = stream;

    blocks_.push_back(new_block);
    total_allocated_ += aligned_size;
    current_usage_ += aligned_size;
    peak_usage_ = std::max(peak_usage_, current_usage_);

    std::cout << "Allocated new GPU memory: " << aligned_size << " bytes (tag: " << tag << ")" << std::endl;
    return ptr;
}

bool GPUMemoryManager::deallocate(void* ptr) {
    std::lock_guard<std::mutex> lock(memory_mutex_);

    for (auto& block : blocks_) {
        if (block.ptr == ptr) {
            if (!block.in_use) {
                std::cerr << "Attempting to deallocate already free memory" << std::endl;
                return false;
            }

            block.in_use = false;
            block.stream = 0;
            current_usage_ -= block.size;

            std::cout << "Deallocated " << block.size << " bytes (tag: " << block.tag << ")" << std::endl;

            // 尝试合并相邻的空闲块
            mergeBlocks();
            return true;
        }
    }

    std::cerr << "Memory block not found for deallocation" << std::endl;
    return false;
}

bool GPUMemoryManager::preallocate(const std::vector<size_t>& sizes, const std::vector<std::string>& tags) {
    std::cout << "Preallocating " << sizes.size() << " memory blocks..." << std::endl;

    for (size_t i = 0; i < sizes.size(); ++i) {
        std::string tag = (i < tags.size()) ? tags[i] : ("preallocated_" + std::to_string(i));
        void* ptr = allocate(sizes[i], tag);
        if (!ptr) {
            std::cerr << "Failed to preallocate block " << i << std::endl;
            return false;
        }
        // 立即释放，但保留在内存池中
        deallocate(ptr);
    }

    std::cout << "Preallocation completed successfully" << std::endl;
    return true;
}

void GPUMemoryManager::cleanup() {
    std::lock_guard<std::mutex> lock(memory_mutex_);

    std::cout << "Cleaning up GPU memory..." << std::endl;

    for (auto& block : blocks_) {
        if (block.ptr) {
            cudaFree(block.ptr);
            block.ptr = nullptr;
        }
    }

    blocks_.clear();
    total_allocated_ = 0;
    current_usage_ = 0;

    std::cout << "GPU memory cleanup completed" << std::endl;
}

void GPUMemoryManager::defragment() {
    std::lock_guard<std::mutex> lock(memory_mutex_);

    std::cout << "Starting memory defragmentation..." << std::endl;

    // 合并所有相邻的空闲块
    mergeBlocks();

    // 按地址排序块
    std::sort(blocks_.begin(), blocks_.end(),
              [](const MemoryBlock& a, const MemoryBlock& b) {
                  return a.ptr < b.ptr;
              });

    std::cout << "Memory defragmentation completed" << std::endl;
}

size_t GPUMemoryManager::getAvailableMemory() const {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
}

void GPUMemoryManager::printMemoryInfo() const {
    std::lock_guard<std::mutex> lock(memory_mutex_);

    std::cout << "\n=== GPU Memory Manager Statistics ===" << std::endl;
    std::cout << "Total allocated: " << total_allocated_ / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Current usage: " << current_usage_ / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Peak usage: " << peak_usage_ / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Available GPU memory: " << getAvailableMemory() / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Number of blocks: " << blocks_.size() << std::endl;

    size_t free_blocks = 0, used_blocks = 0;
    for (const auto& block : blocks_) {
        if (block.in_use) {
            used_blocks++;
        } else {
            free_blocks++;
        }
    }

    std::cout << "Used blocks: " << used_blocks << std::endl;
    std::cout << "Free blocks: " << free_blocks << std::endl;
    std::cout << "====================================\n" << std::endl;
}

bool GPUMemoryManager::checkMemoryLeaks() const {
    std::lock_guard<std::mutex> lock(memory_mutex_);

    size_t leaked_blocks = 0;
    size_t leaked_bytes = 0;

    for (const auto& block : blocks_) {
        if (block.in_use) {
            leaked_blocks++;
            leaked_bytes += block.size;
            std::cout << "Memory leak detected: " << block.size << " bytes (tag: " << block.tag << ")" << std::endl;
        }
    }

    if (leaked_blocks > 0) {
        std::cout << "Total memory leaks: " << leaked_blocks << " blocks, "
                  << leaked_bytes / (1024 * 1024) << " MB" << std::endl;
        return false;
    }

    std::cout << "No memory leaks detected" << std::endl;
    return true;
}

void GPUMemoryManager::resetStats() {
    std::lock_guard<std::mutex> lock(memory_mutex_);
    peak_usage_ = current_usage_;
    std::cout << "Memory statistics reset" << std::endl;
}

void GPUMemoryManager::setMemoryLimit(size_t limit_bytes) {
    std::cout << "Memory limit set to " << limit_bytes / (1024 * 1024) << " MB" << std::endl;
    // 实际实现中可以在分配时检查限制
}

MemoryBlock* GPUMemoryManager::findFreeBlock(size_t size) {
    for (auto& block : blocks_) {
        if (!block.in_use && block.size >= size) {
            return &block;
        }
    }
    return nullptr;
}

void GPUMemoryManager::splitBlock(MemoryBlock* block, size_t size) {
    if (block->size <= size) return;

    size_t remaining_size = block->size - size;
    void* new_ptr = static_cast<char*>(block->ptr) + size;

    MemoryBlock new_block(new_ptr, remaining_size, "split_block");
    blocks_.push_back(new_block);

    block->size = size;
}

void GPUMemoryManager::mergeBlocks() {
    // 按地址排序
    std::sort(blocks_.begin(), blocks_.end(),
              [](const MemoryBlock& a, const MemoryBlock& b) {
                  return a.ptr < b.ptr;
              });

    // 合并相邻的空闲块
    for (size_t i = 0; i < blocks_.size() - 1; ++i) {
        if (!blocks_[i].in_use && !blocks_[i + 1].in_use) {
            char* end_of_first = static_cast<char*>(blocks_[i].ptr) + blocks_[i].size;
            if (end_of_first == blocks_[i + 1].ptr) {
                blocks_[i].size += blocks_[i + 1].size;
                blocks_.erase(blocks_.begin() + i + 1);
                --i; // 重新检查当前位置
            }
        }
    }
}

size_t GPUMemoryManager::alignSize(size_t size) const {
    return ((size + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
}

void GPUMemoryManager::updateUsageStats() {
    // 统计信息在分配/释放时更新
}

// WeightDataManager 实现
WeightDataManager::WeightDataManager(GPUMemoryManager* mem_mgr)
    : memory_manager_(mem_mgr) {
    std::cout << "Weight Data Manager initialized" << std::endl;
}

WeightDataManager::~WeightDataManager() {
    unloadAllWeights();
    std::cout << "Weight Data Manager destroyed" << std::endl;
}

bool WeightDataManager::loadWeights(const std::string& name, const void* host_data, size_t size,
                                   cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(weights_mutex_);

    if (!validateWeightData(host_data, size)) {
        std::cerr << "Invalid weight data for: " << name << std::endl;
        return false;
    }

    // 检查是否已存在
    if (weight_data_.find(name) != weight_data_.end()) {
        std::cout << "Weight already exists, replacing: " << name << std::endl;
        unloadWeights(name);
    }

    // 分配GPU内存
    void* gpu_data = memory_manager_->allocate(size, "weight_" + name, stream);
    if (!gpu_data) {
        std::cerr << "Failed to allocate GPU memory for weights: " << name << std::endl;
        return false;
    }

    // 复制数据到GPU
    cudaError_t error = cudaMemcpy(gpu_data, host_data, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "Failed to copy weights to GPU: " << cudaGetErrorString(error) << std::endl;
        memory_manager_->deallocate(gpu_data);
        return false;
    }

    // 存储权重信息
    weight_data_[name] = gpu_data;
    weight_sizes_[name] = size;
    weight_streams_[name] = stream;

    std::cout << "Weights loaded successfully: " << name << " (" << size << " bytes)" << std::endl;
    return true;
}

void* WeightDataManager::getWeights(const std::string& name) const {
    std::lock_guard<std::mutex> lock(weights_mutex_);
    auto it = weight_data_.find(name);
    return (it != weight_data_.end()) ? it->second : nullptr;
}

size_t WeightDataManager::getWeightSize(const std::string& name) const {
    std::lock_guard<std::mutex> lock(weights_mutex_);
    auto it = weight_sizes_.find(name);
    return (it != weight_sizes_.end()) ? it->second : 0;
}

bool WeightDataManager::hasWeights(const std::string& name) const {
    std::lock_guard<std::mutex> lock(weights_mutex_);
    return weight_data_.find(name) != weight_data_.end();
}

bool WeightDataManager::unloadWeights(const std::string& name) {
    std::lock_guard<std::mutex> lock(weights_mutex_);

    auto it = weight_data_.find(name);
    if (it == weight_data_.end()) {
        std::cerr << "Weights not found: " << name << std::endl;
        return false;
    }

    // 释放GPU内存
    memory_manager_->deallocate(it->second);

    // 移除记录
    weight_data_.erase(it);
    weight_sizes_.erase(name);
    weight_streams_.erase(name);

    std::cout << "Weights unloaded: " << name << std::endl;
    return true;
}

void WeightDataManager::unloadAllWeights() {
    std::lock_guard<std::mutex> lock(weights_mutex_);

    std::cout << "Unloading all weights..." << std::endl;

    for (const auto& pair : weight_data_) {
        memory_manager_->deallocate(pair.second);
    }

    weight_data_.clear();
    weight_sizes_.clear();
    weight_streams_.clear();

    std::cout << "All weights unloaded" << std::endl;
}

bool WeightDataManager::preloadWeights(const std::map<std::string, std::pair<const void*, size_t>>& weights_data) {
    std::cout << "Preloading " << weights_data.size() << " weight tensors..." << std::endl;

    for (const auto& pair : weights_data) {
        if (!loadWeights(pair.first, pair.second.first, pair.second.second)) {
            std::cerr << "Failed to preload weights: " << pair.first << std::endl;
            return false;
        }
    }

    std::cout << "Weight preloading completed successfully" << std::endl;
    return true;
}

std::vector<std::string> WeightDataManager::getWeightNames() const {
    std::lock_guard<std::mutex> lock(weights_mutex_);
    std::vector<std::string> names;
    for (const auto& pair : weight_data_) {
        names.push_back(pair.first);
    }
    return names;
}

size_t WeightDataManager::getTotalWeightSize() const {
    std::lock_guard<std::mutex> lock(weights_mutex_);
    size_t total = 0;
    for (const auto& pair : weight_sizes_) {
        total += pair.second;
    }
    return total;
}

void WeightDataManager::printWeightInfo() const {
    std::lock_guard<std::mutex> lock(weights_mutex_);

    std::cout << "\n=== Weight Data Manager Statistics ===" << std::endl;
    std::cout << "Number of weight tensors: " << weight_data_.size() << std::endl;

    // 直接计算总大小，避免调用getTotalWeightSize()导致死锁
    size_t total_size = 0;
    for (const auto& pair : weight_sizes_) {
        total_size += pair.second;
    }
    std::cout << "Total weight size: " << total_size / (1024 * 1024) << " MB" << std::endl;

    for (const auto& pair : weight_data_) {
        std::cout << "  " << pair.first << ": " << weight_sizes_.at(pair.first) << " bytes" << std::endl;
    }

    std::cout << "======================================\n" << std::endl;
}

bool WeightDataManager::validateWeightData(const void* data, size_t size) const {
    return data != nullptr && size > 0;
}

// TensorMemoryPool 实现
TensorMemoryPool::TensorMemoryPool(GPUMemoryManager* mem_mgr, size_t max_size)
    : memory_manager_(mem_mgr), max_pool_size_(max_size) {
    std::cout << "Tensor Memory Pool initialized with capacity: " << max_size / (1024 * 1024) << " MB" << std::endl;
}

TensorMemoryPool::~TensorMemoryPool() {
    cleanupUnusedTensors();
    std::cout << "Tensor Memory Pool destroyed" << std::endl;
}

void* TensorMemoryPool::allocateTensor(const std::vector<int>& shape, size_t element_size,
                                      const std::string& owner) {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    size_t tensor_size = calculateTensorSize(shape, element_size);

    // 查找合适的槽位
    TensorSlot* slot = findSuitableSlot(shape, element_size);

    if (slot) {
        slot->in_use = true;
        slot->owner = owner;
        std::cout << "Reused tensor slot: " << tensor_size << " bytes (owner: " << owner << ")" << std::endl;
        return slot->ptr;
    }

    // 检查池容量
    if (getPoolUsage() + tensor_size > max_pool_size_) {
        std::cerr << "Tensor pool capacity exceeded" << std::endl;
        return nullptr;
    }

    // 分配新的张量内存
    void* ptr = memory_manager_->allocate(tensor_size, "tensor_" + owner);
    if (!ptr) {
        std::cerr << "Failed to allocate tensor memory" << std::endl;
        return nullptr;
    }

    TensorSlot new_slot;
    new_slot.ptr = ptr;
    new_slot.size = tensor_size;
    new_slot.shape = shape;
    new_slot.in_use = true;
    new_slot.owner = owner;

    tensor_slots_.push_back(new_slot);

    std::cout << "Allocated new tensor: " << tensor_size << " bytes (owner: " << owner << ")" << std::endl;
    return ptr;
}

bool TensorMemoryPool::deallocateTensor(void* ptr) {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    for (auto& slot : tensor_slots_) {
        if (slot.ptr == ptr) {
            if (!slot.in_use) {
                std::cerr << "Attempting to deallocate already free tensor" << std::endl;
                return false;
            }

            slot.in_use = false;
            slot.owner.clear();

            std::cout << "Tensor deallocated: " << slot.size << " bytes" << std::endl;
            return true;
        }
    }

    std::cerr << "Tensor not found for deallocation" << std::endl;
    return false;
}

bool TensorMemoryPool::preallocateTensors(const std::vector<std::pair<std::vector<int>, size_t>>& tensor_specs) {
    std::cout << "Preallocating " << tensor_specs.size() << " tensor slots..." << std::endl;

    for (size_t i = 0; i < tensor_specs.size(); ++i) {
        const auto& spec = tensor_specs[i];
        std::string owner = "preallocated_" + std::to_string(i);

        void* ptr = allocateTensor(spec.first, spec.second, owner);
        if (!ptr) {
            std::cerr << "Failed to preallocate tensor " << i << std::endl;
            return false;
        }

        // 立即释放，但保留在池中
        deallocateTensor(ptr);
    }

    std::cout << "Tensor preallocation completed successfully" << std::endl;
    return true;
}

void TensorMemoryPool::cleanupUnusedTensors() {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    std::cout << "Cleaning up unused tensors..." << std::endl;

    auto it = tensor_slots_.begin();
    while (it != tensor_slots_.end()) {
        if (!it->in_use) {
            memory_manager_->deallocate(it->ptr);
            it = tensor_slots_.erase(it);
        } else {
            ++it;
        }
    }

    std::cout << "Tensor cleanup completed" << std::endl;
}

size_t TensorMemoryPool::getPoolUsage() const {
    size_t usage = 0;
    for (const auto& slot : tensor_slots_) {
        usage += slot.size;
    }
    return usage;
}

void TensorMemoryPool::printPoolInfo() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    std::cout << "\n=== Tensor Memory Pool Statistics ===" << std::endl;
    std::cout << "Pool capacity: " << max_pool_size_ / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Pool usage: " << getPoolUsage() / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Number of tensor slots: " << tensor_slots_.size() << std::endl;

    size_t used_slots = 0, free_slots = 0;
    for (const auto& slot : tensor_slots_) {
        if (slot.in_use) {
            used_slots++;
        } else {
            free_slots++;
        }
    }

    std::cout << "Used slots: " << used_slots << std::endl;
    std::cout << "Free slots: " << free_slots << std::endl;
    std::cout << "====================================\n" << std::endl;
}

TensorMemoryPool::TensorSlot* TensorMemoryPool::findSuitableSlot(const std::vector<int>& shape, size_t element_size) {
    for (auto& slot : tensor_slots_) {
        if (!slot.in_use && shapesMatch(slot.shape, shape)) {
            size_t required_size = calculateTensorSize(shape, element_size);
            if (slot.size >= required_size) {
                return &slot;
            }
        }
    }
    return nullptr;
}

size_t TensorMemoryPool::calculateTensorSize(const std::vector<int>& shape, size_t element_size) const {
    size_t total_elements = 1;
    for (int dim : shape) {
        total_elements *= dim;
    }
    return total_elements * element_size;
}

bool TensorMemoryPool::shapesMatch(const std::vector<int>& shape1, const std::vector<int>& shape2) const {
    return shape1 == shape2;
}

// GlobalMemoryManager 实现
std::unique_ptr<GPUMemoryManager> GlobalMemoryManager::gpu_memory_manager_;
std::unique_ptr<WeightDataManager> GlobalMemoryManager::weight_data_manager_;
std::unique_ptr<TensorMemoryPool> GlobalMemoryManager::tensor_memory_pool_;
std::mutex GlobalMemoryManager::init_mutex_;
bool GlobalMemoryManager::initialized_ = false;

bool GlobalMemoryManager::initialize(size_t gpu_memory_limit, size_t tensor_pool_size) {
    std::lock_guard<std::mutex> lock(init_mutex_);

    if (initialized_) {
        std::cout << "Global Memory Manager already initialized" << std::endl;
        return true;
    }

    std::cout << "Initializing Global Memory Manager..." << std::endl;

    try {
        gpu_memory_manager_ = std::make_unique<GPUMemoryManager>();
        if (gpu_memory_limit > 0) {
            gpu_memory_manager_->setMemoryLimit(gpu_memory_limit);
        }

        weight_data_manager_ = std::make_unique<WeightDataManager>(gpu_memory_manager_.get());
        tensor_memory_pool_ = std::make_unique<TensorMemoryPool>(gpu_memory_manager_.get(), tensor_pool_size);

        initialized_ = true;
        std::cout << "Global Memory Manager initialized successfully" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Global Memory Manager: " << e.what() << std::endl;
        return false;
    }
}

GPUMemoryManager* GlobalMemoryManager::getGPUMemoryManager() {
    if (!initialized_) {
        initialize();
    }
    return gpu_memory_manager_.get();
}

WeightDataManager* GlobalMemoryManager::getWeightDataManager() {
    if (!initialized_) {
        initialize();
    }
    return weight_data_manager_.get();
}

TensorMemoryPool* GlobalMemoryManager::getTensorMemoryPool() {
    if (!initialized_) {
        initialize();
    }
    return tensor_memory_pool_.get();
}

void GlobalMemoryManager::cleanup() {
    std::lock_guard<std::mutex> lock(init_mutex_);

    if (!initialized_) {
        return;
    }

    std::cout << "Cleaning up Global Memory Manager..." << std::endl;

    tensor_memory_pool_.reset();
    weight_data_manager_.reset();
    gpu_memory_manager_.reset();

    initialized_ = false;
    std::cout << "Global Memory Manager cleanup completed" << std::endl;
}

void GlobalMemoryManager::printGlobalMemoryInfo() {
    if (!initialized_) {
        std::cout << "Global Memory Manager not initialized" << std::endl;
        return;
    }

    std::cout << "\n=== Global Memory Manager Statistics ===" << std::endl;
    gpu_memory_manager_->printMemoryInfo();
    weight_data_manager_->printWeightInfo();
    tensor_memory_pool_->printPoolInfo();
    std::cout << "========================================\n" << std::endl;
}

} // namespace yolo
} // namespace cuda_learning
