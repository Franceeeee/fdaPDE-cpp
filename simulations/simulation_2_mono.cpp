
#include "utils.h"

int main(){

    // TEST 2: mono&iter 20000 obs - unit square coarse
    // -- test parameters --
	std::string test_name = "simulation_2/";
	int seed = 23872; 
    double lambda = 1e-3; 
    bool same_locs = 0;
    std::string meshID = "../test/data/mesh/unit_square_coarse/";
    std::size_t m = 3;
	double na_percentage = 0.0;
	DMatrix<double> beta = DMatrix<double>::Zero(2,1);
    beta(0,0) = -2.; beta(1,0) = 1.;
    DMatrix<double> alpha = DMatrix<double>::Zero(3,1);
    alpha(0,0) = -0.5; alpha(1,0) = 0.; alpha(2,0) = 0.5;
    int n_obs = 20000;

    DMatrix<double> nodes = read_csv<double>(meshID + "points.csv");
	DMatrix<int> elements = (read_csv<int>(meshID + "elements.csv").array() - 1).matrix();
	DMatrix<int> boundary = read_csv<int>(meshID + "boundary.csv");
	std::cout << nodes.rows() << " " << nodes.cols() << std::endl;
	Mesh<2,2> mesh = Mesh<2,2>(nodes, elements, boundary);
    
    std::string name_dir = "data/";
	if(!std::filesystem::create_directory(name_dir)) std::filesystem::create_directory(name_dir);

    name_dir += test_name;
    if(!std::filesystem::exists(std::filesystem::path(name_dir))) std::filesystem::create_directory(name_dir);
	
	// input data 
    std::string input_dir = name_dir  + "input/";

    if(!std::filesystem::exists(std::filesystem::path(input_dir))) {

        std::filesystem::create_directory(input_dir);
        
        Eigen::saveMarket(beta, input_dir + "beta.mtx");
        Eigen::saveMarket(alpha, input_dir + "alpha.mtx");
       
        eigen2txt<double>(beta, input_dir + "beta.txt");
        eigen2txt<double>(alpha, input_dir + "alpha.txt");
        
        Eigen::saveMarket(x1_(mesh.nodes()), input_dir + "cov_1.mtx");
        eigen2txt<double>(x1_(mesh.nodes()), input_dir + "cov_1.txt");
        for( std::size_t j=0; j < m; ++j){
            DMatrix<double> f_ = f(mesh.nodes(), j);
            Eigen::saveMarket(f_, input_dir + "f_" + std::to_string(j) + ".mtx");
            eigen2txt<double>(f_, input_dir + "f_" + std::to_string(j) + ".txt");
        }
            
        // generate data
        std::string data_dir = input_dir + std::to_string(n_obs) + "/";
        std::filesystem::create_directory(data_dir);
            
        std::mt19937 gen(seed);
        
        std::string simul_dir = data_dir + "/"; 
        std::filesystem::create_directory(simul_dir);

        for(std::size_t j = 0; j < m; ++j){
        
        	DMatrix<double> locs = uniform_locs(n_obs, gen);
        	
            DMatrix<double> DesignMatrix = DMatrix<double>::Zero(n_obs,2);
        
            DesignMatrix.col(0) = x1_(locs); // va in V
            DesignMatrix.col(1) = noise(n_obs, 1.0, gen);

            DMatrix<double> f_ = f(locs, j);
            double sigma = 0.05*std::abs(f_.array().maxCoeff() - f_.array().minCoeff()); 
            auto eps_ = noise(n_obs, sigma, gen);
            eigen2txt<double>(eps_, simul_dir + "noise_" + std::to_string(j) + ".txt");
        
            DMatrix<double> obs = DesignMatrix * beta + DesignMatrix.col(0)*alpha(j,0)  + f_ + eps_; 
            
            auto na_mask = create_na_mask(n_obs, na_percentage, gen); 
            for (int i = 0; i < n_obs; ++i) { 
                if (na_mask[i]) {
                    obs(i, 0) = std::numeric_limits<double>::quiet_NaN();  
                }
            }
        
            Eigen::saveMarket(locs, simul_dir + "locs_" + std::to_string(j) + ".mtx");
            Eigen::saveMarket(DesignMatrix, simul_dir + "DesignMatrix_" + std::to_string(j) + ".mtx");
            Eigen::saveMarket(DesignMatrix.col(1), simul_dir + "W_" + std::to_string(j) + ".mtx");
            Eigen::saveMarket(DesignMatrix.col(0), simul_dir + "V_" + std::to_string(j) + ".mtx");
            Eigen::saveMarket(obs, simul_dir + "obs_" + std::to_string(j) + ".mtx");

            eigen2txt<double>(locs, simul_dir + "locs_" + std::to_string(j) + ".txt");
            eigen2txt<double>(DesignMatrix, simul_dir + "DesignMatrix_" + std::to_string(j) + ".txt");
            eigen2txt<double>(DesignMatrix.col(1), simul_dir + "W_" + std::to_string(j) + ".txt");
            eigen2txt<double>(DesignMatrix.col(0), simul_dir + "V_" + std::to_string(j) + ".txt");
            eigen2txt<double>(obs, simul_dir + "obs_" + std::to_string(j) + ".txt");
        }
        
    }

	// output directory
	std::string output_dir = name_dir + "output/";
    if(!std::filesystem::exists(std::filesystem::path(output_dir))){ 
        std::filesystem::create_directory(output_dir);
    }

    // import data from files
    
    output_dir = name_dir + "output/"; 
    
    std::string data_dir = input_dir + std::to_string(n_obs) + "/";
    if(!std::filesystem::exists(std::filesystem::path(output_dir))) std::filesystem::create_directory(output_dir);

    std::string simul_dir = data_dir + "/"; 
    std::string result_dir = output_dir + "/";
    if(!std::filesystem::exists(std::filesystem::path(result_dir))) std::filesystem::create_directory(result_dir);

    std::string output_monolithic = result_dir + "monolithic/";
    if(!std::filesystem::exists(std::filesystem::path(output_monolithic))) std::filesystem::create_directory(output_monolithic);
    
    std::vector<BlockFrame<double, int>> data;
    data.resize(m);
    
    // insert data in blockframes
    for(std::size_t j = 0; j<m; j++){
        std::string Wname = simul_dir + "W_" + std::to_string(j) + ".mtx";
        std::string Vname = simul_dir + "V_" + std::to_string(j) + ".mtx";
        std::string locsname = simul_dir + "locs_" + std::to_string(j) + ".mtx";
        std::string yname = simul_dir +  "obs_" + std::to_string(j) + ".mtx";
        auto W = read_mtx<double>(Wname);
        auto V = read_mtx<double>(Vname);
        auto locs = read_mtx<double>(locsname);
        auto obs = read_mtx<double>(yname);

        data[j].insert(W_BLOCK, W);
        data[j].insert(V_BLOCK, V);
        data[j].insert(Y_BLOCK, obs);
        data[j].insert(LOCS_BLOCK, locs);      
    }

    DMatrix<double> f_ = DMatrix<double>::Zero(m*mesh.nodes().rows(),1);
    for(std::size_t j = 0; j < m; ++j){
        f_.block(j*mesh.nodes().rows(),0, mesh.nodes().rows(),1) =   f(mesh.nodes(),j);
    }

    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(mesh.n_elements()*3, 1);
    PDE<decltype(mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(mesh, L, u);

    // monolithic 
    fANOVA<monolithic> monolithic_(problem, Sampling::pointwise, same_locs);
    monolithic_.set_lambda_D(lambda);
    monolithic_.set_data(data);
    
    monolithic_.init();
    
    monolithic_.solve();
        
    for(std::size_t j = 0; j < m; ++j){
        eigen2txt<double>(monolithic_.f().block(j*mesh.nodes().rows(),0, mesh.nodes().rows(),1),
                        output_monolithic + "estimate_f_" + std::to_string(j) + ".txt");
    }
    
    eigen2txt<double>(monolithic_.beta(), output_monolithic + "beta.txt");
    eigen2txt<double>(monolithic_.alpha(), output_monolithic + "alpha.txt");
    
	return 0;
}
