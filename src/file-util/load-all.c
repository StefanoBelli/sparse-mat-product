#include <file-util/load-all.h>

const struct matrix_file_info mandatory[] = {
    { "vanHeukelum", "cage4" },
    { "Bai",         "mhda416" },
    { "HB",          "mcfe" },
    { "Bai",         "olm1000" },
    { "Sandia",      "adder_dcop_32" },
    { "HB",          "west2021" },
    { "DRIVCAV",     "cavity10" },
    { "Zitney",      "rdist2" },
    { "Williams",    "cant" },
    { "Simon",       "olafu" },
    { "Janna",       "Cube_Coup_dt0" },
    { "Janna",       "ML_Laplace" },
    { "HB",          "bcsstk17" },
    { "Williams",    "mac_econ_fwd500" },
    { "Bai",         "mhd4800a" },
    { "Williams",    "cop20k_A" },
    { "Simon",       "raefsky2" },
    { "Bai",         "af23560" },
    { "Norris",      "lung2" },
    { "Fluorem",     "PR02R" },
    { "Botonakis",   "FEM_3D_thermal1" },
    { "Schmid",      "thermal1" },
    { "Schmid",      "thermal2" },
    { "Botonakis",   "thermomech_TK" },
    { "Schenk",      "nlpkkt80" },
    { "Williams",    "webbase-1M" },
    { "IBM_EDA",     "dc1" },
    { "SNAP",        "amazon0302" },
    { "Schenk_AFE",  "af_1_k101" },
    { "SNAP",        "roadNet-PA" }
};

struct matrix_loaded_file 
*load_all_files(
    const char *basedir, 
    const struct matrix_file_info *additionals, 
    int num_more, 
    int *loaded_num) 
{

    
    
}