//#include <string.h>
//#include <string>
//#include <fstream>
//#include <filesystem>
//#include <iostream>

struct COOStorage
{
    int arow;
    int acol;
    float aval;
};
//
//struct CSVRecord{
//    std::string matrix_name;
//    std::string matrix_type;
//    std::string device;
//    int rows;
//    int cols;
//    int nnz;
//    float flops;
//    float bandwidth;
//    float geometric_mean;
//    float aritmetic_mean;
//    float std_dev;
//
//
//
//};
//
//class csvLogHandler{
//    csvLogHandler(std::filesystem::path filepath) : file_(std::string(filepath)) {
//        // Checking if parent directory exists
//        if(!std::filesystem::is_directory(filepath.parent_path())){
//           std::filesystem::create_directory(filepath.parent_path());
//        }
//
//
//        if(!file_.is_open()){
//            std::cout << "Something has gone wrong while opening log file " << std::endl;
//            return;
//        }
//
//        file_ << "matrix_name,matrix_type,device,rows,cols,nnz,flops,bandwidth,geometric_mean,aritmetic_mean,std_dev\n";
//        file_.flush();
//    }
//
//    void logResults(std::string matrix_name, std::string matrix_type, std::string device,
//                    int rows, int cols, int nnz, float flops, float bandwidth, float geometric_mean,
//                    float aritmetic_mean, float std_dev){
//
//        file_ << matrix_name << "," <<
//            matrix_name << "," <<
//            matrix_type << "," <<
//            device << "," <<
//            rows << "," <<
//            cols << "," <<
//            nnz << "," <<
//            flops << "," <<
//            bandwidth << "," <<
//            geometric_mean << "," <<
//            aritmetic_mean << "," <<
//            std_dev << ",";
//
//        file_.flush();
//
//    }
//    ~csvLogHandler(){
//        file_.close();
//    }
//
//
//
//
//    private:
//        std::ofstream file_;
//};
