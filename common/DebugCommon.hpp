#pragma once
#include <iostream>
#include <string>

namespace common {

// Get section string
std::string section(const std::string& title, bool breakLine = true) {
    std::string str;
    if (breakLine) {
        str += "\n";
    }
    if (title.empty()) {
        return str + std::string(120, '=');
    }

    std::string fillStr((120 - title.size()) / 2, '=');
    return str + fillStr + " " + title + " " + fillStr;
}

// Get sub section string
std::string subSection(const std::string& title, bool breakLine = true) {
    std::string str;
    if (breakLine) {
        str += "\n";
    }
    if (title.empty()) {
        return str + std::string(120, '-');
    }

    std::string fillStr((120 - title.size()) / 2, '-');
    return str + fillStr + " " + title + " " + fillStr;
}

}  // namespace common