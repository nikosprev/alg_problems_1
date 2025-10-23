#include <iostream>
#include <string>
#include <vector>
#include <cstdlib> // For system()

// Function to download a file from a URL 
bool downloadFile(const std::string& url, const std::string& outputFilename) {
    
    std::string command = "curl -L \"" + url + "\" -o \"" + outputFilename + "\"";  // CUrl command

    std::cout << "Attempting to download: " << url << std::endl;
    std::cout << "Saving to: " << outputFilename << std::endl;
    std::cout << "Executing command: " << command << std::endl;

    // Execute the command
    int result = std::system(command.c_str());

    if (result == 0) {
        std::cout << "Successfully downloaded " << outputFilename << std::endl;
        return true;
    } else {
        std::cerr << "Error downloading " << outputFilename << ". Curl exit code: " << result << std::endl;
     
        return false;
    }
}

