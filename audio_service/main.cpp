#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main() {
    std::ifstream file("../protocol/schemas.json");
    json schemas;
    file >> schemas;
}