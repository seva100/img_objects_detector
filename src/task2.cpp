#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>
#include <utility>

#include "classifier.h"
#include "matrix.h"
#include "EasyBMP.h"
#include "EasyBMP_DataStructures.h"
#include "linear.h"
#include "argvparser.h"

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;

using CommandLineProcessing::ArgvParser;

typedef vector<pair<BMP*, int> > TDataSet;
typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<float>, int> > TFeatures;

const double PI = 3.14159265359;

// Load list of files and its labels from 'data_file' and
// stores it in 'file_list'
void LoadFileList(const string& data_file, TFileList* file_list) {
    ifstream stream(data_file.c_str());

    string filename;
    int label;
    
    int char_idx = data_file.size() - 1;
    for (; char_idx >= 0; --char_idx)
        if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
            break;
    string data_path = data_file.substr(0,char_idx+1);
    
    while(!stream.eof() && !stream.fail()) {
        stream >> filename >> label;
        if (filename.size())
            file_list->push_back(make_pair(data_path + filename, label));
    }

    stream.close();
}

// Load images by list of files 'file_list' and store them in 'data_set'
void LoadImages(const TFileList& file_list, TDataSet* data_set) {
    for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
            // Create image
        BMP* image = new BMP();
            // Read image from file
        image->ReadFromFile(file_list[img_idx].first.c_str());
            // Add image and it's label to dataset
        data_set->push_back(make_pair(image, file_list[img_idx].second));
    }
}

// Save result of prediction to file
void SavePredictions(const TFileList& file_list,
                     const TLabels& labels, 
                     const string& prediction_file) {
        // Check that list of files and list of labels has equal size 
    assert(file_list.size() == labels.size());
        // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());

        // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
        stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
    stream.close();
}

template <class T>
void init_matrix(Matrix<T> &m) {
    for (int i = 0; i < int(m.n_rows); ++i) {
        for (int j = 0; j < int(m.n_cols); ++j) {
            m(i, j) = 0;
        }
    }
}

Matrix<int> to_grayscale(BMP &image) {
    int width = image.TellWidth();
    int height = image.TellHeight();
    Matrix<int> grayscale(width, height);
    init_matrix(grayscale);
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            RGBApixel pixel = image.GetPixel(i, j);
            grayscale(i, j) = .299 *  pixel.Red + .587 * pixel.Green + .114 * pixel.Blue;
        }
    }
    return grayscale;
}

int convolve_in_pos(const Matrix<int> &kernel, const Matrix<int> &image,
    const pair<int, int> &pos) {
    int row_rad = kernel.n_rows / 2;
    int col_rad = kernel.n_cols / 2;
    int conv = 0;
    for (int i = pos.first - row_rad; i <= pos.first + row_rad; ++i) {
        for (int j = pos.second - col_rad; j <= pos.second + col_rad; ++j) {
            conv += kernel(i - (pos.first - row_rad), j - (pos.second - col_rad)) *
                    image(i, j);
        }
    }
    return conv;
}

Matrix<int> convolve(const Matrix<int> &kernel, const Matrix<int> &image) {
    if ((kernel.n_rows % 2 == 0) ||
        (kernel.n_cols % 2 == 0)) {
        throw "Wrong size of kernel";
    }
    int row_rad = kernel.n_rows / 2;
    int col_rad = kernel.n_cols / 2;
    Matrix<int> conv(image.n_rows - 2 * row_rad,
                     image.n_cols - 2 * col_rad);
    init_matrix(conv);
    pair<int, int> pos(0, 0);
    for (int i = row_rad + 1; i < int(image.n_rows) - row_rad; ++i) {
        for (int j = col_rad + 1; j < int(image.n_cols) - col_rad; ++j) {
            pos.first = i;
            pos.second = j;
            conv(i - row_rad - 1, j - col_rad - 1) = 
                convolve_in_pos(kernel, image, pos);
        }
    }
    return conv;
}

void save_grayscale(const Matrix<int> &image, const string &output_file)
{
    // Debug function.
    // Saves grayscale image `image` into file named `output_file`.
    BMP res;
    res.SetSize(int(image.n_rows), int(image.n_cols));
    for (int i = 0; i < int(image.n_rows); ++i) {
        for (int j = 0; j < int(image.n_cols); ++j) {
            RGBApixel pixel;
            pixel.Red = pixel.Green = pixel.Blue = image(i, j);
            pixel.Alpha = 0;
            res.SetPixel(i, j, pixel);
        }
    }
    res.WriteToFile(output_file.c_str());
}

vector<float> compute_hog(const Matrix<double> &grad,
                          const Matrix<double> &grad_dir,
                          int n_folds) {
    vector<float> hog(n_folds, 0);
    double fold_len = 2 * PI / double(n_folds);
    for (int i = 0; i < int(grad.n_rows); ++i) {
        for (int j = 0; j < int(grad.n_cols); ++j) {
            int cur_fold = int((grad_dir(i, j) + PI) / fold_len) % n_folds;
            //cout << int((grad_dir(i, j) + PI) / fold_len) << ' ';
            hog[cur_fold] += float(grad(i, j));
        }
    }
    //cout << endl;
    return hog;
}

inline bool is_close(double first, double second, double a_tol=1e-7) {
    return (fabs(first - second) <= a_tol);
}

double angle(double y, double x) {
    if (is_close(x, 0)) {
        if (is_close(y, 0)) {
            // In this case it doesn't matter what angle to choose,
            // since histogram will receive zero addition
            return 0;
        } else {
            if (y > 0) {
                return PI / 2.0;
            }
            return -PI / 2.0;
        }
    } 
    return atan2(y, x);
}

vector<float> normalize_hog(const vector<float> &hog) {
    vector<float> norm_hog(int(hog.size()));
    double norm = 0;
    for (int i = 0; i < int(hog.size()); ++i) {
        norm += hog[i] * hog[i];
    }
    if (is_close(norm, 0)) {
        return hog;
    }
    norm = sqrt(norm);
    for (int i = 0; i < int(hog.size()); ++i) {
        norm_hog[i] = hog[i] / norm;
    }
    return norm_hog;
}

void
check_for_valid_cell(int i, int j, int rows, int cols, const string &comment) {
    if (!(0 <= i && i < rows && 0 <= j && j < cols)) {
        cout << "WRONG CELL: " << comment << " (" << i << ", " << j << ") in " << 
            "(" << rows << ", " << cols << ")" << endl;
        cout.flush();
    }
}

template <class T>
inline void extend_vector(vector<T> &first, vector<T> second) {
    first.reserve(int(first.size()) + distance(second.begin(), second.end()));
    first.insert(first.end(), second.begin(), second.end());
}

vector<float> hog_descr(const Matrix<double> &grad, const Matrix<double> &grad_dir) {

    // Computing HOG for each cell and concatenating them in descriptor.
    vector<float> descr;
    // N_CELLS: how many cells for each side we divide image into.
    // Total number of cells will be N_CELLS * N_CELLS.
    const int N_CELLS = 4;
    // N_FOLDS: number of bins in HOG.
    const int N_FOLDS = 8;

    int cell_rows = int(double(grad.n_rows) / N_CELLS);
    int cell_cols = int(double(grad.n_cols) / N_CELLS);
    // Second, add HOG for each cell, ignoring the residuals 
    // (2nd part of descriptors pyramid).
    Matrix<vector<float> > hog(N_CELLS, N_CELLS);

    for (int cell_i = 0; cell_i < N_CELLS; ++cell_i) {
        for (int cell_j = 0; cell_j < N_CELLS; ++cell_j) {
            int i = cell_i * cell_rows;
            int j = cell_j * cell_cols;
            Matrix<double> cell_grad = grad.submatrix(i, j, cell_rows, cell_cols);
            Matrix<double> cell_grad_dir = 
                grad_dir.submatrix(i, j, cell_rows, cell_cols);

            /*if (cell_i == 0 && cell_j == 0) {
                cout << "cell_grad: " << endl;
                cout << cell_grad << endl;
                cout << "cell_grad_dir: " << endl;
                cout << cell_grad_dir << endl;
            }*/

            //check_for_valid_cell(cell_i, cell_j, N_CELLS, N_CELLS, "hog1");
            hog(cell_i, cell_j) = compute_hog(cell_grad, cell_grad_dir, N_FOLDS);
            /*if (image_idx == 1 && cell_i == 0 && cell_j == 0) {
                for (int k = 0; k < int(hog(cell_i, cell_j).size()); ++k) {
                    cout << hog(cell_i, cell_j)[k] << ' ';
                } cout << endl;
            }*/
            hog(cell_i, cell_j) = normalize_hog(hog(cell_i, cell_j));
            // Appending this hog to descriptor
            extend_vector(descr, hog(cell_i, cell_j));
        }
    }
    return descr;
}

vector<float> color_features(const BMP &image, int width, int height) {

    // N_CELLS: how many cells for each side we divide image into
    // to calculate color features in each cell.
    // Total number of cells will be N_CELLS * N_CELLS.
    const int N_CELLS = 8;
    vector<float> features;

    int cell_rows = width / N_CELLS;
    int cell_cols = height / N_CELLS;
    for (int cell_i = 0; cell_i < N_CELLS; ++cell_i) {
        for (int cell_j = 0; cell_j < N_CELLS; ++cell_j) {
            float mean_color[3] = {0, 0, 0};
            int start_i = cell_i * cell_rows;
            int end_i = (cell_i + 1) * cell_rows;
            int start_j = cell_j * cell_cols;
            int end_j = (cell_j + 1) * cell_cols;
            for (int i = start_i; i < end_i; ++i) {
                for (int j = start_j; j < end_j; ++j) {
                    RGBApixel pixel = image.GetPixel(i, j);
                    mean_color[0] += pixel.Red;
                    mean_color[1] += pixel.Green;
                    mean_color[2] += pixel.Blue;
                }
            }
            // Dividing sum by quantity and by 255 in order to 
            // transform [0, 255] range to [0.0, 1.0].
            for (int k = 0; k < 3; ++k) {
                mean_color[k] /= cell_rows * cell_cols * 255;
                features.push_back(mean_color[k]);
            }
        }
    }
    return features;
}

// Extract features from dataset.
// You should implement this function by yourself =)
void ExtractFeatures(const TDataSet& data_set, TFeatures* features) {
    for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx) {
        int width = data_set[image_idx].first->TellWidth();
        int height = data_set[image_idx].first->TellHeight();

        // Transforming image to grayscale.
        Matrix<int> grayscale = to_grayscale(*data_set[image_idx].first);

        // Computing convolution with horizontal and vertical Sobel filters.
        Matrix<int> Sobel_hor = {{-1, 0, 1}};
        //Matrix<int> Sobel_ver = { {1}, {0}, {-1} };
        Matrix<int> Sobel_ver(3, 1);
        Sobel_ver(0, 0) = -1;
        Sobel_ver(1, 0) = 0;
        Sobel_ver(2, 0) = 1;
        //cout << "grayscale shape: " << int(grayscale.n_rows) << ' ' << int(grayscale.n_cols) << endl;
        Matrix<int> conv_hor = convolve(Sobel_hor, grayscale); // m x (n-2)
        Matrix<int> conv_ver = convolve(Sobel_ver, grayscale); // (m-2) x n
        //cout << "conv_hor shape: " << int(conv_hor.n_rows) << ' ' << int(conv_hor.n_cols) << endl;
        //cout << "conv_ver shape: " << int(conv_ver.n_rows) << ' ' << int(conv_ver.n_cols) << endl;
        //save_grayscale(conv_hor, "Conv hor.bmp");
        //save_grayscale(conv_ver, "Conv ver.bmp");

        // Computing gradient and its direction.
        // Ignoring border of size 1px.
        int new_width = width - 2;
        int new_height = height - 2;
        Matrix<double> grad(new_width, new_height);
        Matrix<double> grad_dir(new_width, new_height);
        for (int i = 0; i < new_width; ++i) {
            for (int j = 0; j < new_height; ++j) {
                grad(i, j) = sqrt(double(conv_hor(i + 1, j) * conv_hor(i + 1, j)) +
                                  double(conv_ver(i, j + 1) * conv_ver(i, j + 1)));
                grad_dir(i, j) = angle(double(-conv_ver(i, j + 1)), 
                                       double(conv_hor(i + 1, j)));
            }
        }

        vector<float> descr;
        // Creating descriptor based on pyramid of descriptors.
        // N_CELLS_PYR: how many cells for each side we divide image into
        // to compute pyramid of descriptors.
        // Total number of cells will be N_CELLS_PYR * N_CELLS_PYR.
        const int N_CELLS_PYR = 2;
        int cell_rows = int(double(new_width / N_CELLS_PYR));
        int cell_cols = int(double(new_height / N_CELLS_PYR));
        extend_vector(descr, hog_descr(grad, grad_dir));
        for (int cell_i = 0; cell_i < N_CELLS_PYR; ++cell_i) {
            for (int cell_j = 0; cell_j < N_CELLS_PYR; ++cell_j) {
                Matrix<double> cell_grad = grad.submatrix(cell_i * cell_rows,
                    cell_j * cell_cols, cell_rows, cell_cols);
                Matrix<double> cell_grad_dir = grad_dir.submatrix(cell_i * cell_rows,
                    cell_j * cell_cols, cell_rows, cell_cols);
                extend_vector(descr, hog_descr(cell_grad, cell_grad_dir));
            }
        }
        extend_vector(descr, color_features(*data_set[image_idx].first, width, height));

        /*if (image_idx == 1) {
            for (int i = 0; i < int(descr.size()); ++i) {
                cout << descr[i] << ' ';
            }
            cout << endl;
        }*/

        // Transforming features according to approximate chi-squared kernel formula
        // Parameters: 
        // CHI_SQ_N: approximation order n
        const int CHI_SQ_N = 2;
        // CHI_SQ_L: approximation step
        const float CHI_SQ_L = 2.0;
        vector<float> descr_chi;
        for (float x: descr) {
            for (float lambda = -CHI_SQ_N * CHI_SQ_L; 
                lambda <= CHI_SQ_N * CHI_SQ_L;
                lambda += CHI_SQ_L) {
                // sech(x) = 1 / cosh(x)
                float real_part;
                float img_part;
                if (is_close(x, 0)) {
                    real_part = img_part = 0;
                } else {
                    float common_part = sqrt(x / cosh(lambda));
                    real_part = cos(lambda * log(x)) * common_part;
                    img_part = sin(-lambda * log(x)) * common_part;
                }
                descr_chi.push_back(real_part);
                descr_chi.push_back(img_part);
            }
        }

        //features->push_back(make_pair(descr, data_set[image_idx].second));
        features->push_back(make_pair(descr_chi, data_set[image_idx].second));

        // Remove this sample code and place your feature extraction code here
        /*vector<float> one_image_features;
        one_image_features.push_back(1.0);
        features->push_back(make_pair(one_image_features, 1));*/
        // End of sample code
    }
}

// Clear dataset structure
void ClearDataset(TDataSet* data_set) {
        // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
        // Clear dataset
    data_set->clear();
}

// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
void TrainClassifier(const string& data_file, const string& model_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // Model which would be trained
    TModel model;
        // Parameters of classifier
    TClassifierParams params;
    
        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // PLACE YOUR CODE HERE
        // You can change parameters of classifier here
    params.C = 0.01;
    TClassifier classifier(params);
        // Train classifier
    classifier.Train(features, &model);
        // Save model to file
    model.Save(model_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

// Predict data from 'data_file' using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string& data_file,
                 const string& model_file,
                 const string& prediction_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // List of image labels
    TLabels labels;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // Classifier 
    TClassifier classifier = TClassifier(TClassifierParams());
        // Trained model
    TModel model;
        // Load model from file
    model.Load(model_file);
        // Predict images by its features using 'model' and store predictions
        // to 'labels'
    classifier.Predict(features, model, &labels);

        // Save predictions
    SavePredictions(file_list, labels, prediction_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

int main(int argc, char** argv) {
    // Command line options parser
    ArgvParser cmd;
        // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2015.");
        // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
        // Add other options
    cmd.defineOption("data_set", "File with dataset",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("model", "Path to file to save or load model",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
        ArgvParser::OptionRequiresValue);
    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");
        
        // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");

        // Parse options
    int result = cmd.parse(argc, argv);

        // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }

        // Get values 
    string data_file = cmd.optionValue("data_set");
    string model_file = cmd.optionValue("model");
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");

        // If we need to train classifier
    if (train)
        TrainClassifier(data_file, model_file);
        // If we need to predict data
    if (predict) {
            // You must declare file to save images
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
            // File to save predictions
        string prediction_file = cmd.optionValue("predicted_labels");
            // Predict data
        PredictData(data_file, model_file, prediction_file);
    }
}