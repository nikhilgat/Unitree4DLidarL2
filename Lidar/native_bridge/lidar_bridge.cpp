// Minimal native relay: talks to the Unitree LiDAR over UDP using the vendor
// SDK, and re-broadcasts each parsed point cloud frame to any number of
// local TCP clients (e.g. the visualize/save Python scripts).
//
// Wire format per frame, all little-endian (native x86_64):
//   4 bytes  magic "PCLD"
//   8 bytes  double stamp
//   4 bytes  uint32 num_points (N)
//   N * 16 bytes  float32 x, y, z, intensity

#include "unitree_lidar_sdk.h"

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <signal.h>
#include <cstring>
#include <cstdio>
#include <vector>
#include <mutex>
#include <thread>

using namespace unilidar_sdk2;

static std::vector<int> g_clients;
static std::mutex g_clients_mutex;

static void acceptLoop(int server_fd)
{
    while (true)
    {
        int client_fd = accept(server_fd, nullptr, nullptr);
        if (client_fd < 0) continue;
        std::lock_guard<std::mutex> lock(g_clients_mutex);
        g_clients.push_back(client_fd);
        printf("Client connected (fd=%d), total clients=%zu\n", client_fd, g_clients.size());
    }
}

static void broadcastCloud(const PointCloudUnitree &cloud)
{
    std::lock_guard<std::mutex> lock(g_clients_mutex);
    if (g_clients.empty()) return;

    uint32_t n = (uint32_t)cloud.points.size();
    std::vector<uint8_t> buf(4 + 8 + 4 + (size_t)n * 16);
    size_t off = 0;
    memcpy(&buf[off], "PCLD", 4); off += 4;
    memcpy(&buf[off], &cloud.stamp, 8); off += 8;
    memcpy(&buf[off], &n, 4); off += 4;
    for (uint32_t i = 0; i < n; i++)
    {
        float vals[4] = {cloud.points[i].x, cloud.points[i].y, cloud.points[i].z, cloud.points[i].intensity};
        memcpy(&buf[off], vals, 16);
        off += 16;
    }

    for (auto it = g_clients.begin(); it != g_clients.end(); )
    {
        ssize_t sent = send(*it, buf.data(), buf.size(), MSG_NOSIGNAL);
        if (sent < 0)
        {
            close(*it);
            it = g_clients.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

int main()
{
    signal(SIGPIPE, SIG_IGN);

    UnitreeLidarReader *lreader = createUnitreeLidarReader();

    const std::string lidar_ip = "192.168.1.62";
    const std::string local_ip = "192.168.1.2";
    const unsigned short lidar_port = 6101;
    const unsigned short local_port = 6201;

    if (lreader->initializeUDP(lidar_port, lidar_ip, local_port, local_ip))
    {
        printf("Unilidar UDP initialization failed! Exit here.\n");
        return -1;
    }
    printf("Unilidar UDP initialization succeeded.\n");

    lreader->startLidarRotation();
    sleep(1);
    lreader->setLidarWorkMode(0);
    sleep(1);

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    addr.sin_port = htons(9899);
    if (bind(server_fd, (sockaddr *)&addr, sizeof(addr)) < 0)
    {
        printf("Failed to bind relay TCP port 9899. Is another instance already running?\n");
        return -1;
    }
    listen(server_fd, 4);
    printf("Relay listening on 127.0.0.1:9899\n");

    std::thread(acceptLoop, server_fd).detach();

    PointCloudUnitree cloud;
    while (true)
    {
        int result = lreader->runParse();
        if (result == LIDAR_POINT_DATA_PACKET_TYPE)
        {
            if (lreader->getPointCloud(cloud))
            {
                broadcastCloud(cloud);
            }
        }
    }
    return 0;
}
