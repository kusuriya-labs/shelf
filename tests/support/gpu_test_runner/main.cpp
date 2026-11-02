#include <sys/file.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using clk = std::chrono::steady_clock;

namespace {

constexpr long kFallbackBudgetGb = 24;

struct Config {
    int gpu_count = 1;
    long cost_gb = 1;
    long timeout_sec = 120;
    long reserve_timeout_sec = 900;
    long budget_gb = 0;
    std::string state_dir;
    std::vector<std::string> cmd;
};

Config g_cfg;
pid_t g_child = -1;
bool g_released = false;

[[noreturn]] void fail(const std::string& msg) {
    std::fprintf(stderr, "gpu_test_runner: %s\n", msg.c_str());
    std::exit(1);
}

long env_long(const char* name, long def) {
    const char* v = std::getenv(name);
    if (!v || !*v) return def;
    char* end = nullptr;
    long r = std::strtol(v, &end, 10);
    return end == v ? def : r;
}

int open_lock() {
    int fd = ::open((g_cfg.state_dir + "/lock").c_str(), O_RDWR | O_CREAT, 0644);
    if (fd >= 0 && flock(fd, LOCK_EX) != 0) {
        ::close(fd);
        return -1;
    }
    return fd;
}

void close_lock(int fd) {
    if (fd >= 0) {
        flock(fd, LOCK_UN);
        ::close(fd);
    }
}

std::vector<std::string> render_nodes() {
    std::vector<std::string> out;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator("/sys/class/drm", ec)) {
        const std::string n = e.path().filename().string();
        if (n.rfind("renderD", 0) != 0 || n.size() <= 7) continue;
        bool digits = true;
        for (char c : n.substr(7)) {
            if (c < '0' || c > '9') {
                digits = false;
                break;
            }
        }
        if (digits) out.push_back(n);
    }
    std::sort(out.begin(), out.end());
    return out;
}

bool integrated_render_node(const std::string& node) {
    std::error_code ec;
    return fs::exists("/sys/class/drm/" + node + "/device/firmware_node", ec);
}

long vram_budget_gb(int gpu) {
    if (g_cfg.budget_gb > 0) return g_cfg.budget_gb;
    static const std::vector<std::string> nodes = render_nodes();
    if (gpu < 0 || static_cast<size_t>(gpu) >= nodes.size()) {
        return kFallbackBudgetGb;
    }
    std::ifstream in("/sys/class/drm/" + nodes[gpu] +
                     "/device/mem_info_vram_total");
    long long bytes = 0;
    if (!(in >> bytes) || bytes <= 0) return kFallbackBudgetGb;
    return static_cast<long>(bytes / (1024ll * 1024 * 1024));
}

std::vector<std::tuple<int, int, long>> read_entries() {
    std::vector<std::tuple<int, int, long>> out;
    std::ifstream f(g_cfg.state_dir + "/usage");
    std::string line;
    while (std::getline(f, line)) {
        int pid = 0;
        int gpu = 0;
        long cost = 0;
        if (std::sscanf(line.c_str(), "%d %d %ld", &pid, &gpu, &cost) != 3) continue;
        if (pid > 0 && kill(static_cast<pid_t>(pid), 0) == 0) out.emplace_back(pid, gpu, cost);
    }
    return out;
}

void write_entries(const std::vector<std::tuple<int, int, long>>& entries) {
    std::string tmp = g_cfg.state_dir + "/usage.tmp";
    std::ofstream f(tmp);
    for (const auto& [pid, gpu, cost] : entries) f << pid << ' ' << gpu << ' ' << cost << '\n';
    f.close();
    fs::rename(tmp, g_cfg.state_dir + "/usage");
}

long wall_sec() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

std::vector<std::pair<int, long>> read_taint() {
    std::vector<std::pair<int, long>> out;
    std::ifstream f(g_cfg.state_dir + "/taint");
    std::string line;
    while (std::getline(f, line)) {
        int gpu = 0;
        long expiry = 0;
        if (std::sscanf(line.c_str(), "%d %ld", &gpu, &expiry) == 2) out.emplace_back(gpu, expiry);
    }
    return out;
}

void write_taint(const std::vector<std::pair<int, long>>& t) {
    std::string tmp = g_cfg.state_dir + "/taint.tmp";
    std::ofstream f(tmp);
    for (const auto& [g, e] : t) f << g << ' ' << e << '\n';
    f.close();
    fs::rename(tmp, g_cfg.state_dir + "/taint");
}

void taint_gpus(const std::vector<int>& gpus, long ttl_sec) {
    int fd = open_lock();
    if (fd < 0) return;
    long now = wall_sec();
    long expiry = now + ttl_sec;
    auto t = read_taint();
    for (int g : gpus) {
        bool found = false;
        for (auto& e : t) {
            if (e.first == g) {
                if (e.second < expiry) e.second = expiry;
                found = true;
                break;
            }
        }
        if (!found) t.emplace_back(g, expiry);
    }
    t.erase(std::remove_if(t.begin(), t.end(), [now](const auto& e) { return e.second <= now; }), t.end());
    write_taint(t);
    close_lock(fd);
}

void release() {
    if (g_released) return;
    g_released = true;
    int fd = open_lock();
    if (fd < 0) return;
    auto entries = read_entries();
    int me = static_cast<int>(getpid());
    std::erase_if(entries, [me](const auto& e) { return std::get<0>(e) == me; });
    write_entries(entries);
    close_lock(fd);
}

std::vector<int> candidate_gpus() {
    std::vector<int> out;
    const char* src = std::getenv("PHASESHIFT_TEST_GPUS");
    if (!src || !*src) src = std::getenv("HIP_VISIBLE_DEVICES");
    if (src && *src) {
        std::string s(src);
        std::string cur;
        for (char c : s) {
            if (c == ',') {
                if (!cur.empty()) out.push_back(std::atoi(cur.c_str()));
                cur.clear();
            } else if (c != ' ') {
                cur.push_back(c);
            }
        }
        if (!cur.empty()) out.push_back(std::atoi(cur.c_str()));
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }
    const std::vector<std::string> nodes = render_nodes();
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (!integrated_render_node(nodes[i])) out.push_back(static_cast<int>(i));
    }
    if (!out.empty()) return out;
    for (size_t i = 0; i < nodes.size(); ++i) out.push_back(static_cast<int>(i));
    return out;
}

bool reserve(const std::vector<int>& candidates, std::vector<int>* picked) {
    auto deadline = clk::now() + std::chrono::seconds(g_cfg.reserve_timeout_sec);
    for (;;) {
        int fd = open_lock();
        if (fd >= 0) {
            auto entries = read_entries();
            std::vector<long> used(64, 0);
            for (const auto& [pid, gpu, cost] : entries) {
                while (static_cast<int>(used.size()) <= gpu) used.push_back(0);
                used[gpu] += cost;
            }
            long now = wall_sec();
            auto taint = read_taint();
            std::vector<int> fit;
            for (int g : candidates) {
                while (static_cast<int>(used.size()) <= g) used.push_back(0);
                bool bad = false;
                for (const auto& [tg, te] : taint) {
                    if (tg == g && te > now) {
                        bad = true;
                        break;
                    }
                }
                if (!bad && used[g] + g_cfg.cost_gb <= vram_budget_gb(g)) {
                    fit.push_back(g);
                }
            }
            if (static_cast<int>(fit.size()) >= g_cfg.gpu_count) {
                std::stable_sort(fit.begin(), fit.end(),
                                 [&used](int a, int b) {
                                     if (used[a] != used[b]) return used[a] < used[b];
                                     return a < b;
                                 });
                picked->assign(fit.begin(), fit.begin() + g_cfg.gpu_count);
                for (int g : *picked) entries.emplace_back(static_cast<int>(getpid()), g, g_cfg.cost_gb);
                write_entries(entries);
                close_lock(fd);
                return true;
            }
            close_lock(fd);
        }
        if (clk::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

void kill_child_group() {
    if (g_child > 0) {
        kill(static_cast<pid_t>(-g_child), SIGKILL);
        kill(g_child, SIGKILL);
    }
}

void on_signal(int) {
    kill_child_group();
    _exit(1);
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> args(argv, argv + argc);
    bool seen_dd = false;
    for (std::size_t i = 1; i < args.size(); ++i) {
        const auto& a = args[i];
        if (seen_dd) {
            g_cfg.cmd.push_back(a);
            continue;
        }
        if (a == "--") {
            seen_dd = true;
            continue;
        }
        if (i + 1 >= args.size()) fail("missing value for " + a);
        const char* v = args[i + 1].c_str();
        if (a == "--gpu-count") g_cfg.gpu_count = std::atoi(v);
        else if (a == "--cost-gb") g_cfg.cost_gb = std::atol(v);
        else if (a == "--timeout") g_cfg.timeout_sec = std::atol(v);
        else if (a == "--reserve-timeout") g_cfg.reserve_timeout_sec = std::atol(v);
        else if (a == "--budget-gb") g_cfg.budget_gb = std::atol(v);
        else if (a == "--state-dir") g_cfg.state_dir = v;
        else fail("unknown arg: " + a);
        ++i;
    }
    if (g_cfg.cmd.empty()) fail("no command");
    if (g_cfg.state_dir.empty()) fail("missing --state-dir");
    g_cfg.budget_gb = env_long("PHASESHIFT_TEST_GPU_BUDGET_GB", g_cfg.budget_gb);
    g_cfg.reserve_timeout_sec = env_long("PHASESHIFT_TEST_GPU_RESERVE_TIMEOUT", g_cfg.reserve_timeout_sec);

    std::error_code ec;
    fs::create_directories(g_cfg.state_dir, ec);

    auto candidates = candidate_gpus();
    if (static_cast<int>(candidates.size()) < g_cfg.gpu_count) {
        std::printf("SKIP: insufficient GPUs (need %d, have %zu)\n", g_cfg.gpu_count, candidates.size());
        return 77;
    }

    std::vector<int> picked;
    if (!reserve(candidates, &picked)) {
        std::fprintf(stderr, "FAIL: could not reserve GPU within %lds\n", g_cfg.reserve_timeout_sec);
        return 1;
    }

    std::string vis;
    for (std::size_t i = 0; i < picked.size(); ++i) {
        if (i) vis += ",";
        vis += std::to_string(picked[i]);
    }
    char vis_buf[256];
    std::snprintf(vis_buf, sizeof(vis_buf), "%s", vis.c_str());

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    std::atexit(release);

    std::vector<char*> child_argv(g_cfg.cmd.size() + 1, nullptr);
    for (std::size_t i = 0; i < g_cfg.cmd.size(); ++i) child_argv[i] = const_cast<char*>(g_cfg.cmd[i].c_str());

    g_child = fork();
    if (g_child < 0) fail("fork");
    if (g_child == 0) {
        if (setpgid(0, 0) != 0) _exit(127);
        setenv("HIP_VISIBLE_DEVICES", vis_buf, 1);
        execvp(g_cfg.cmd[0].c_str(), child_argv.data());
        _exit(127);
    }

    auto deadline = clk::now() + std::chrono::seconds(g_cfg.timeout_sec);
    int rc = 1;
    for (;;) {
        int st = 0;
        pid_t r = waitpid(g_child, &st, WNOHANG);
        if (r == g_child) {
            if (WIFEXITED(st)) rc = WEXITSTATUS(st);
            break;
        }
        if (r < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (clk::now() >= deadline) {
            std::fprintf(stderr, "FAIL: test process did not exit within %lds; SIGKILL\n", g_cfg.timeout_sec);
            kill_child_group();
            taint_gpus(picked, 300);
            for (int i = 0; i < 500; ++i) {
                int wst = 0;
                if (waitpid(g_child, &wst, WNOHANG) == g_child) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            release();
            return 1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    release();
    if (rc == 0) return 0;
    if (rc == 77) return 77;
    std::fprintf(stderr, "gpu_test_runner: child exit %d\n", rc);
    return 1;
}
