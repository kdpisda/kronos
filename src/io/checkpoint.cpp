// ============================================================================
// KRONOS  src/io/checkpoint.cpp
// Checkpoint/restart implementation.
//
// Two backends:
//   1. HDF5 (if KRONOS_HAS_HDF5 is defined)
//   2. Portable binary fallback with header:
//        Magic  "KCKP" (4 bytes)
//        Version uint32_t = 1
//        Input hash: 32 bytes (zero-padded)
//        SCF step: int32_t
//        Then length-prefixed arrays of data.
//
// All writes are atomic: write to <filename>.tmp, then rename.
// ============================================================================

#include "io/checkpoint.hpp"
#include "utils/logger.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#ifdef KRONOS_HAS_HDF5
#include <hdf5.h>
#endif

namespace kronos::checkpoint {

// ============================================================================
// Input hash: DJB2 variant producing 64-bit hash, output as 16-char hex
// ============================================================================

std::string compute_input_hash(const std::string& yaml_content) {
    // DJB2a (xor variant) — fast, good distribution for text
    uint64_t hash = 5381;
    for (unsigned char c : yaml_content) {
        hash = ((hash << 5) + hash) ^ c;  // hash * 33 ^ c
    }

    // Format as 16-char lowercase hex
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx",
                  static_cast<unsigned long long>(hash));
    return std::string(buf);
}

// ============================================================================
// Binary format helpers
// ============================================================================

namespace {

constexpr char MAGIC[4] = {'K', 'C', 'K', 'P'};
constexpr uint32_t FORMAT_VERSION = 1;
constexpr size_t HASH_FIELD_SIZE = 32;  // fixed-size field for input hash

// Write a POD value
template <typename T>
void write_pod(std::ostream& os, const T& val) {
    os.write(reinterpret_cast<const char*>(&val), sizeof(T));
}

// Read a POD value
template <typename T>
void read_pod(std::istream& is, T& val) {
    is.read(reinterpret_cast<char*>(&val), sizeof(T));
}

// Write a vector of POD
template <typename T>
void write_vector(std::ostream& os, const std::vector<T>& vec) {
    uint64_t sz = vec.size();
    write_pod(os, sz);
    if (sz > 0) {
        os.write(reinterpret_cast<const char*>(vec.data()),
                 static_cast<std::streamsize>(sz * sizeof(T)));
    }
}

// Read a vector of POD
template <typename T>
void read_vector(std::istream& is, std::vector<T>& vec) {
    uint64_t sz = 0;
    read_pod(is, sz);
    vec.resize(static_cast<size_t>(sz));
    if (sz > 0) {
        is.read(reinterpret_cast<char*>(vec.data()),
                static_cast<std::streamsize>(sz * sizeof(T)));
    }
}

// Write a vector of vectors of POD
template <typename T>
void write_vector_of_vectors(std::ostream& os,
                             const std::vector<std::vector<T>>& vv) {
    uint64_t outer = vv.size();
    write_pod(os, outer);
    for (const auto& v : vv) {
        write_vector(os, v);
    }
}

// Read a vector of vectors of POD
template <typename T>
void read_vector_of_vectors(std::istream& is,
                            std::vector<std::vector<T>>& vv) {
    uint64_t outer = 0;
    read_pod(is, outer);
    vv.resize(static_cast<size_t>(outer));
    for (auto& v : vv) {
        read_vector(is, v);
    }
}

// ============================================================================
// Binary write
// ============================================================================

void write_binary(const std::string& path, const CheckpointData& data) {
    std::ofstream os(path, std::ios::binary);
    if (!os.is_open()) {
        throw std::runtime_error("checkpoint: failed to open " + path +
                                 " for writing");
    }

    // Header
    os.write(MAGIC, 4);
    write_pod(os, FORMAT_VERSION);

    // Input hash (fixed 32 bytes, zero-padded)
    char hash_buf[HASH_FIELD_SIZE] = {};
    std::strncpy(hash_buf, data.input_hash.c_str(), HASH_FIELD_SIZE - 1);
    os.write(hash_buf, HASH_FIELD_SIZE);

    // SCF step
    int32_t step32 = static_cast<int32_t>(data.scf_step);
    write_pod(os, step32);

    // Density (complex)
    write_vector(os, data.density_g);

    // Wavefunctions: vector of vector<complex>
    write_vector_of_vectors(os, data.wavefunctions);

    // Eigenvalues: vector of vector<double>
    write_vector_of_vectors(os, data.eigenvalues);

    // Occupations: vector of vector<double>
    write_vector_of_vectors(os, data.occupations);

    // DIIS history: vector of vector<complex>
    write_vector_of_vectors(os, data.diis_density_history);

    os.flush();
    if (!os.good()) {
        throw std::runtime_error("checkpoint: write error on " + path);
    }
}

// ============================================================================
// Binary read
// ============================================================================

CheckpointData read_binary(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is.is_open()) {
        throw std::runtime_error("checkpoint: failed to open " + path +
                                 " for reading");
    }

    // Verify magic
    char magic[4];
    is.read(magic, 4);
    if (std::memcmp(magic, MAGIC, 4) != 0) {
        throw std::runtime_error(
            "checkpoint: invalid magic number in " + path +
            " (expected KCKP)");
    }

    // Verify version
    uint32_t version = 0;
    read_pod(is, version);
    if (version != FORMAT_VERSION) {
        throw std::runtime_error(
            "checkpoint: unsupported version " + std::to_string(version) +
            " in " + path + " (expected " +
            std::to_string(FORMAT_VERSION) + ")");
    }

    CheckpointData data;

    // Input hash
    char hash_buf[HASH_FIELD_SIZE] = {};
    is.read(hash_buf, HASH_FIELD_SIZE);
    data.input_hash = std::string(hash_buf);

    // SCF step
    int32_t step32 = 0;
    read_pod(is, step32);
    data.scf_step = static_cast<int>(step32);

    // Density
    read_vector(is, data.density_g);

    // Wavefunctions
    read_vector_of_vectors(is, data.wavefunctions);

    // Eigenvalues
    read_vector_of_vectors(is, data.eigenvalues);

    // Occupations
    read_vector_of_vectors(is, data.occupations);

    // DIIS history
    read_vector_of_vectors(is, data.diis_density_history);

    if (!is.good() && !is.eof()) {
        throw std::runtime_error("checkpoint: read error on " + path);
    }

    return data;
}

} // anonymous namespace

// ============================================================================
// HDF5 backend
// ============================================================================

#ifdef KRONOS_HAS_HDF5

namespace {

// Helper: write a 1D complex-double dataset
void hdf5_write_complex_1d(hid_t loc, const char* name,
                           const std::vector<std::complex<double>>& vec) {
    hsize_t dims[1] = {vec.size() * 2};  // store as interleaved real/imag
    hid_t space = H5Screate_simple(1, dims, nullptr);
    hid_t dset = H5Dcreate2(loc, name, H5T_NATIVE_DOUBLE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // complex<double> is layout-compatible with double[2]
    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
             reinterpret_cast<const double*>(vec.data()));
    H5Dclose(dset);
    H5Sclose(space);
}

// Helper: read a 1D complex-double dataset
std::vector<std::complex<double>> hdf5_read_complex_1d(hid_t loc,
                                                        const char* name) {
    hid_t dset = H5Dopen2(loc, name, H5P_DEFAULT);
    if (dset < 0) return {};
    hid_t space = H5Dget_space(dset);
    hsize_t dims[1] = {0};
    H5Sget_simple_extent_dims(space, dims, nullptr);
    size_t n_doubles = static_cast<size_t>(dims[0]);
    size_t n_complex = n_doubles / 2;
    std::vector<std::complex<double>> vec(n_complex);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
            reinterpret_cast<double*>(vec.data()));
    H5Sclose(space);
    H5Dclose(dset);
    return vec;
}

// Helper: write a 1D double dataset
void hdf5_write_double_1d(hid_t loc, const char* name,
                          const std::vector<double>& vec) {
    hsize_t dims[1] = {vec.size()};
    hid_t space = H5Screate_simple(1, dims, nullptr);
    hid_t dset = H5Dcreate2(loc, name, H5T_NATIVE_DOUBLE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
             vec.data());
    H5Dclose(dset);
    H5Sclose(space);
}

// Helper: read a 1D double dataset
std::vector<double> hdf5_read_double_1d(hid_t loc, const char* name) {
    hid_t dset = H5Dopen2(loc, name, H5P_DEFAULT);
    if (dset < 0) return {};
    hid_t space = H5Dget_space(dset);
    hsize_t dims[1] = {0};
    H5Sget_simple_extent_dims(space, dims, nullptr);
    std::vector<double> vec(static_cast<size_t>(dims[0]));
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
            vec.data());
    H5Sclose(space);
    H5Dclose(dset);
    return vec;
}

// Helper: write a scalar string attribute
void hdf5_write_string_attr(hid_t loc, const char* name,
                            const std::string& val) {
    hid_t atype = H5Tcopy(H5T_C_S1);
    H5Tset_size(atype, val.size() + 1);
    H5Tset_strpad(atype, H5T_STR_NULLTERM);
    hid_t aspace = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(loc, name, atype, aspace,
                            H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, atype, val.c_str());
    H5Aclose(attr);
    H5Sclose(aspace);
    H5Tclose(atype);
}

// Helper: read a scalar string attribute
std::string hdf5_read_string_attr(hid_t loc, const char* name) {
    hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
    if (attr < 0) return "";
    hid_t atype = H5Aget_type(attr);
    size_t sz = H5Tget_size(atype);
    std::string val(sz, '\0');
    hid_t mtype = H5Tcopy(H5T_C_S1);
    H5Tset_size(mtype, sz);
    H5Aread(attr, mtype, val.data());
    H5Tclose(mtype);
    H5Tclose(atype);
    H5Aclose(attr);
    // Trim null terminator
    while (!val.empty() && val.back() == '\0') val.pop_back();
    return val;
}

// Helper: write a scalar int attribute
void hdf5_write_int_attr(hid_t loc, const char* name, int val) {
    hid_t aspace = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(loc, name, H5T_NATIVE_INT, aspace,
                            H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_INT, &val);
    H5Aclose(attr);
    H5Sclose(aspace);
}

// Helper: read a scalar int attribute
int hdf5_read_int_attr(hid_t loc, const char* name) {
    hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
    if (attr < 0) return 0;
    int val = 0;
    H5Aread(attr, H5T_NATIVE_INT, &val);
    H5Aclose(attr);
    return val;
}

void write_hdf5(const std::string& path, const CheckpointData& data) {
    hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                           H5P_DEFAULT);
    if (file < 0) {
        throw std::runtime_error("checkpoint: failed to create HDF5 file " +
                                 path);
    }

    // Root attributes
    hdf5_write_string_attr(file, "input_hash", data.input_hash);
    hdf5_write_int_attr(file, "scf_step", data.scf_step);
    hdf5_write_string_attr(file, "format", "kronos_checkpoint_v1");

    // Density
    hdf5_write_complex_1d(file, "density_g", data.density_g);

    // Wavefunctions group
    {
        hid_t grp = H5Gcreate2(file, "wavefunctions", H5P_DEFAULT,
                                H5P_DEFAULT, H5P_DEFAULT);
        hdf5_write_int_attr(grp, "num_kpoints",
                            static_cast<int>(data.wavefunctions.size()));
        for (size_t ik = 0; ik < data.wavefunctions.size(); ++ik) {
            std::string dname = "kpoint_" + std::to_string(ik);
            hdf5_write_complex_1d(grp, dname.c_str(), data.wavefunctions[ik]);
        }
        H5Gclose(grp);
    }

    // Eigenvalues group
    {
        hid_t grp = H5Gcreate2(file, "eigenvalues", H5P_DEFAULT,
                                H5P_DEFAULT, H5P_DEFAULT);
        hdf5_write_int_attr(grp, "num_kpoints",
                            static_cast<int>(data.eigenvalues.size()));
        for (size_t ik = 0; ik < data.eigenvalues.size(); ++ik) {
            std::string dname = "kpoint_" + std::to_string(ik);
            hdf5_write_double_1d(grp, dname.c_str(), data.eigenvalues[ik]);
        }
        H5Gclose(grp);
    }

    // Occupations group
    {
        hid_t grp = H5Gcreate2(file, "occupations", H5P_DEFAULT,
                                H5P_DEFAULT, H5P_DEFAULT);
        hdf5_write_int_attr(grp, "num_kpoints",
                            static_cast<int>(data.occupations.size()));
        for (size_t ik = 0; ik < data.occupations.size(); ++ik) {
            std::string dname = "kpoint_" + std::to_string(ik);
            hdf5_write_double_1d(grp, dname.c_str(), data.occupations[ik]);
        }
        H5Gclose(grp);
    }

    // DIIS history group
    {
        hid_t grp = H5Gcreate2(file, "diis_history", H5P_DEFAULT,
                                H5P_DEFAULT, H5P_DEFAULT);
        hdf5_write_int_attr(grp, "num_entries",
                            static_cast<int>(data.diis_density_history.size()));
        for (size_t i = 0; i < data.diis_density_history.size(); ++i) {
            std::string dname = "entry_" + std::to_string(i);
            hdf5_write_complex_1d(grp, dname.c_str(),
                                  data.diis_density_history[i]);
        }
        H5Gclose(grp);
    }

    H5Fclose(file);
}

CheckpointData read_hdf5(const std::string& path) {
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        throw std::runtime_error("checkpoint: failed to open HDF5 file " +
                                 path);
    }

    CheckpointData data;
    data.input_hash = hdf5_read_string_attr(file, "input_hash");
    data.scf_step = hdf5_read_int_attr(file, "scf_step");

    // Density
    data.density_g = hdf5_read_complex_1d(file, "density_g");

    // Wavefunctions
    {
        hid_t grp = H5Gopen2(file, "wavefunctions", H5P_DEFAULT);
        if (grp >= 0) {
            int nk = hdf5_read_int_attr(grp, "num_kpoints");
            data.wavefunctions.resize(nk);
            for (int ik = 0; ik < nk; ++ik) {
                std::string dname = "kpoint_" + std::to_string(ik);
                data.wavefunctions[ik] =
                    hdf5_read_complex_1d(grp, dname.c_str());
            }
            H5Gclose(grp);
        }
    }

    // Eigenvalues
    {
        hid_t grp = H5Gopen2(file, "eigenvalues", H5P_DEFAULT);
        if (grp >= 0) {
            int nk = hdf5_read_int_attr(grp, "num_kpoints");
            data.eigenvalues.resize(nk);
            for (int ik = 0; ik < nk; ++ik) {
                std::string dname = "kpoint_" + std::to_string(ik);
                data.eigenvalues[ik] =
                    hdf5_read_double_1d(grp, dname.c_str());
            }
            H5Gclose(grp);
        }
    }

    // Occupations
    {
        hid_t grp = H5Gopen2(file, "occupations", H5P_DEFAULT);
        if (grp >= 0) {
            int nk = hdf5_read_int_attr(grp, "num_kpoints");
            data.occupations.resize(nk);
            for (int ik = 0; ik < nk; ++ik) {
                std::string dname = "kpoint_" + std::to_string(ik);
                data.occupations[ik] =
                    hdf5_read_double_1d(grp, dname.c_str());
            }
            H5Gclose(grp);
        }
    }

    // DIIS history
    {
        hid_t grp = H5Gopen2(file, "diis_history", H5P_DEFAULT);
        if (grp >= 0) {
            int n = hdf5_read_int_attr(grp, "num_entries");
            data.diis_density_history.resize(n);
            for (int i = 0; i < n; ++i) {
                std::string dname = "entry_" + std::to_string(i);
                data.diis_density_history[i] =
                    hdf5_read_complex_1d(grp, dname.c_str());
            }
            H5Gclose(grp);
        }
    }

    H5Fclose(file);
    return data;
}

} // anonymous namespace

#endif // KRONOS_HAS_HDF5

// ============================================================================
// Public API
// ============================================================================

bool checkpoint_exists(const std::string& filename) {
    return std::filesystem::exists(filename) &&
           std::filesystem::file_size(filename) > 0;
}

void write_checkpoint(const std::string& filename,
                      const CheckpointData& data) {
    auto& logger = Logger::instance();
    std::string tmp_path = filename + ".tmp";

    try {
#ifdef KRONOS_HAS_HDF5
        // Prefer HDF5 if the filename ends with .h5 or .hdf5
        bool use_hdf5 = (filename.size() >= 3 &&
                         (filename.substr(filename.size() - 3) == ".h5" ||
                          filename.find(".hdf5") != std::string::npos));
        if (use_hdf5) {
            write_hdf5(tmp_path, data);
        } else {
            write_binary(tmp_path, data);
        }
#else
        write_binary(tmp_path, data);
#endif

        // Atomic rename
        std::filesystem::rename(tmp_path, filename);

        logger.info("checkpoint", "Checkpoint written",
            {{"file", filename},
             {"scf_step", std::to_string(data.scf_step)},
             {"density_size", std::to_string(data.density_g.size())},
             {"num_kpoints", std::to_string(data.wavefunctions.size())}});

    } catch (...) {
        // Clean up temp file on failure
        std::filesystem::remove(tmp_path);
        throw;
    }
}

CheckpointData read_checkpoint(const std::string& filename) {
    auto& logger = Logger::instance();

    if (!checkpoint_exists(filename)) {
        throw std::runtime_error("checkpoint: file not found: " + filename);
    }

    CheckpointData data;

#ifdef KRONOS_HAS_HDF5
    // Detect format: check if the file starts with the HDF5 magic number
    // (0x89 0x48 0x44 0x46) or the KCKP magic.
    {
        std::ifstream probe(filename, std::ios::binary);
        char header[4] = {};
        probe.read(header, 4);
        if (std::memcmp(header, MAGIC, 4) == 0) {
            // Binary format
            data = read_binary(filename);
        } else {
            // Assume HDF5
            data = read_hdf5(filename);
        }
    }
#else
    data = read_binary(filename);
#endif

    logger.info("checkpoint", "Checkpoint loaded",
        {{"file", filename},
         {"scf_step", std::to_string(data.scf_step)},
         {"input_hash", data.input_hash},
         {"density_size", std::to_string(data.density_g.size())},
         {"num_kpoints", std::to_string(data.wavefunctions.size())}});

    return data;
}

} // namespace kronos::checkpoint
