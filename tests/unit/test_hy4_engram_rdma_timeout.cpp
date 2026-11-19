#include <shelf/models/hy4_flash/rdma/client.h>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    try {
        shelf::hy4::rdma::Client client;
        shelf::hy4::rdma::Indices indices;
        indices[0].push_back(0);
        setenv("SHELF_TEST_TIMEOUT", "1", 1);
        try {
            client.read(indices);
            throw std::runtime_error("タイムアウトしませんでした");
        } catch (const std::exception &e) {
            if (std::string(e.what()).find("タイムアウト") == std::string::npos)
                throw;
            std::cout << e.what() << '\n';
        }
        unsetenv("SHELF_TEST_TIMEOUT");
        try {
            client.read(indices);
            throw std::runtime_error("失敗済み接続で再度 READ できました");
        } catch (const std::exception &e) {
            if (std::string(e.what()).find("失敗済み") == std::string::npos)
                throw;
            std::cout << e.what() << '\n';
        }
        client.shutdown();
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
