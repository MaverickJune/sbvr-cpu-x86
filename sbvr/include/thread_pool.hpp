// // thread_pool.hpp  ─────────────────────────────────────────────
// #pragma once
// #include <thread>
// #include <vector>
// #include <functional>
// #include <condition_variable>
// #include <atomic>

// class ThreadPool {
// public:
//     void init(int n_threads)
//     {
//         if (ready_) return;

//         quit_          = false;
//         pending_tasks_ = 0;

//         if (n_threads <= 0) {
//             unsigned int hw = std::thread::hardware_concurrency();
//             n_threads = hw ? static_cast<int>(hw) : 4;
//         }

//         // print n_threads
//         printf("ThreadPool: using %d threads\n", n_threads);

//         workers_.reserve(n_threads);

//         for (int id = 0; id < n_threads; ++id) {
//             workers_.emplace_back([this, id] { thread_loop(id); });
//         }
//         ready_ = true;
//     }

//     int num_threads() const noexcept
//     {
//         return static_cast<int>(workers_.size());
//     }

//     template<typename Fn>
//     void parallel_for(int n_tasks, Fn&& fn)
//     {
//         if (!ready_ || n_tasks <= 0) {
//             for (int i = 0; i < n_tasks; ++i) fn(i);
//             return;
//         }

//         {
//             std::unique_lock lk(m_);
//             task_fn_       = std::function<void(int)>(std::forward<Fn>(fn));
//             n_tasks_       = n_tasks;
//             next_task_     = 0;
//             pending_tasks_ = n_tasks;
//         }
//         cv_start_.notify_all();

//         std::unique_lock lk(m_done_);
//         cv_done_.wait(lk, [&]{ return pending_tasks_ == 0; });
//     }

//     void finalize()
//     {
//         if (!ready_) return;
//         {
//             std::lock_guard lk(m_);
//             quit_ = true;
//         }
//         cv_start_.notify_all();
//         for (auto& th : workers_) th.join();
//         workers_.clear();
//         ready_ = false;
//     }

//     ~ThreadPool() { finalize(); }

// private:
//     void thread_loop(int /*id*/)
//     {
//         while (true) {
//             std::function<void(int)>* fn = nullptr;
//             int task_id = -1;

//             {
//                 std::unique_lock lk(m_);
//                 cv_start_.wait(lk, [&]{
//                     return quit_ || next_task_ < n_tasks_;
//                 });
//                 if (quit_) return;
//                 task_id = next_task_++;
//             }

//             task_fn_(task_id);

//             if (--pending_tasks_ == 0) {
//                 std::lock_guard lk(m_done_);        // 같은 락으로 보호
//                 cv_done_.notify_one();
//             }
//         }
//     }

//     std::vector<std::thread>          workers_;

//     std::mutex                        m_;
//     std::condition_variable           cv_start_;

//     std::mutex                        m_done_;
//     std::condition_variable           cv_done_;

//     std::function<void(int)>          task_fn_;
//     int                               n_tasks_{0};
//     int                               next_task_{0};

//     std::atomic<int>                  pending_tasks_{0};
//     bool                              quit_{false};
//     bool                              ready_{false};
// };

// Old version of thread_pool.hpp above (before refactoring for better code organization)

// thread_pool.hpp  ─────────────────────────────────────────────
#pragma once
#include <thread>
#include <vector>
#include <functional>
#include <condition_variable>
#include <atomic>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

class ThreadPool {
public:
    void init(int n_threads)
    {
        if (ready_) return;

        quit_          = false;
        pending_tasks_ = 0;

        if (n_threads <= 0) {
            unsigned int hw = std::thread::hardware_concurrency();
            n_threads = hw ? static_cast<int>(hw) : 4;
        }

        printf("ThreadPool: using %d threads\n", n_threads);

        workers_.reserve(n_threads);

        for (int id = 0; id < n_threads; ++id) {
            workers_.emplace_back([this, id] { thread_loop(id); });
        }

#ifdef __linux__
        // Pin each worker thread to a distinct CPU core
        for (int id = 0; id < n_threads; ++id) {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(id, &cpuset);
            int rc = pthread_setaffinity_np(
                workers_[id].native_handle(),
                sizeof(cpu_set_t),
                &cpuset);
            if (rc != 0) {
                fprintf(stderr,
                    "ThreadPool: warning: failed to pin thread %d to core %d (rc=%d)\n",
                    id, id, rc);
            }
        }
#endif

        ready_ = true;
    }

    int num_threads() const noexcept
    {
        return static_cast<int>(workers_.size());
    }

    template<typename Fn>
    void parallel_for(int n_tasks, Fn&& fn)
    {
        if (!ready_ || n_tasks <= 0) {
            for (int i = 0; i < n_tasks; ++i) fn(i);
            return;
        }

        {
            std::unique_lock lk(m_);
            task_fn_       = std::function<void(int)>(std::forward<Fn>(fn));
            n_tasks_       = n_tasks;
            next_task_     = 0;
            pending_tasks_ = n_tasks;
        }
        cv_start_.notify_all();

        std::unique_lock lk(m_done_);
        cv_done_.wait(lk, [&]{ return pending_tasks_ == 0; });
    }

    void finalize()
    {
        if (!ready_) return;
        {
            std::lock_guard lk(m_);
            quit_ = true;
        }
        cv_start_.notify_all();
        for (auto& th : workers_) th.join();
        workers_.clear();
        ready_ = false;
    }

    ~ThreadPool() { finalize(); }

private:
    void thread_loop(int /*id*/)
    {
        while (true) {
            int task_id = -1;

            {
                std::unique_lock lk(m_);
                cv_start_.wait(lk, [&]{
                    return quit_ || next_task_ < n_tasks_;
                });
                if (quit_) return;
                task_id = next_task_++;
            }

            task_fn_(task_id);

            if (--pending_tasks_ == 0) {
                std::lock_guard lk(m_done_);
                cv_done_.notify_one();
            }
        }
    }

    std::vector<std::thread>          workers_;

    std::mutex                        m_;
    std::condition_variable           cv_start_;

    std::mutex                        m_done_;
    std::condition_variable           cv_done_;

    std::function<void(int)>          task_fn_;
    int                               n_tasks_{0};
    int                               next_task_{0};

    std::atomic<int>                  pending_tasks_{0};
    bool                              quit_{false};
    bool                              ready_{false};
};