import sys
import csv
import statistics

avg = lambda x : sum(x) / len(x)
var = lambda x : statistics.variance(x)

def find_underscore_from_end(s, n):
    indices = [i for i, char in enumerate(s) if char == '_']
    
    if len(indices) < n:
        return None
    
    return indices[-n]

gpu_data_aggregate = {}
ser_data_aggregate = {}
mt_data_aggregate = {}

mtxs_with_nonzeroes = {}
measurements_path = sys.argv[1]

with open(f"{measurements_path}/results/nonzeroes.csv", "r") as nzscsv:
    nzs = csv.reader(nzscsv)
    nzs.__next__()
    for row in nzs:
        mtxs_with_nonzeroes[row[0]] = int(row[1])

mtxnzs_latex = []

for mtx in mtxs_with_nonzeroes:
    mtxnzs_latex.append(f"\t\t\t\\textbf{{{mtx.replace('_','\\_')}}} & {mtxs_with_nonzeroes[mtx]} \\\\")

for mtx_name in mtxs_with_nonzeroes:
    fname_serial = f"{measurements_path}/results/{mtx_name}_serial.csv"
    fname_cpu_mt = f"{measurements_path}/results/{mtx_name}_cpu_mt.csv"
    fname_gpu = f"{measurements_path}/results/{mtx_name}_gpu.csv"

    with open(fname_gpu, "r") as gpucsv:
        gpu_res = csv.reader(gpucsv)
        gpu_res.__next__()
        for row in gpu_res:
            for mtx_fmt in [ 'csr', 'hll' ]:
                for kernel_vers in [ 'v1', 'v2', 'v3' ]:
                    if row[1] == mtx_fmt and row[2] == kernel_vers:
                        try:
                            gpu_data_aggregate[f"{mtx_name}_{mtx_fmt}_{kernel_vers}"].append(float(row[6]))
                        except KeyError:
                            gpu_data_aggregate[f"{mtx_name}_{mtx_fmt}_{kernel_vers}"] = [ float(row[6]) ]

    with open(fname_serial, "r") as sercsv:
        ser_res = csv.reader(sercsv)
        ser_res.__next__()
        for row in ser_res:
            for mtx_fmt in [ 'csr', 'hll' ]:
                if row[1] == mtx_fmt:
                    try:
                        ser_data_aggregate[f"{mtx_name}_{mtx_fmt}_v1"].append(float(row[6]))
                    except KeyError:
                        ser_data_aggregate[f"{mtx_name}_{mtx_fmt}_v1"] = [ float(row[6]) ]

    with open(fname_cpu_mt, "r") as mtcsv:
        mt_res = csv.reader(mtcsv)
        mt_res.__next__()
        for row in mt_res:
            for mtx_fmt in [ 'csr', 'hll' ]:
                if row[1] == mtx_fmt and int(row[5]) > 1:
                    try:
                        mt_data_aggregate[f"{mtx_name}_{mtx_fmt}_{row[5]}_v1"].append(float(row[6]))
                    except KeyError:
                        mt_data_aggregate[f"{mtx_name}_{mtx_fmt}_{row[5]}_v1"] = [ float(row[6]) ]   

DEFAULT_CAPTION = "Min, max, avg e var. del tempo d'esecuzione, GFLOPS"

diz ={
        "csr_v1_gpu":["CSR v1 GPU", DEFAULT_CAPTION,[]],
        "csr_v2_gpu":["CSR v2 GPU", DEFAULT_CAPTION,[]],
        "csr_v3_gpu":["CSR v3 GPU", DEFAULT_CAPTION,[]],
        "hll_v1_gpu":["HLL v1 GPU", DEFAULT_CAPTION,[]],
        "hll_v2_gpu":["HLL v2 GPU", DEFAULT_CAPTION,[]],
        "csr_v1_ser":["CSR v1 1 thread CPU",DEFAULT_CAPTION,[]],
        "hll_v1_ser":["HLL v1 1 thread CPU",DEFAULT_CAPTION,[]],
}
for i in range(2, 41):
    diz[f"csr_{i}_v1_mt"] = [f"CSR v1 {i} threads CPU",DEFAULT_CAPTION,[]]
    diz[f"hll_{i}_v1_mt"] = [f"HLL v1 {i} threads CPU",DEFAULT_CAPTION,[]]
 
for k in diz.keys():
    for gpu in gpu_data_aggregate:
        mtxname = gpu[:find_underscore_from_end(gpu, 2)]
        min_t = min(gpu_data_aggregate[gpu])
        max_t = max(gpu_data_aggregate[gpu])
        avg_t = avg(gpu_data_aggregate[gpu])
        var_t = var(gpu_data_aggregate[gpu])
        gflops = (2 * mtxs_with_nonzeroes[mtxname] / avg_t) / 1e9
        if f"{gpu}_gpu".endswith(k):
            diz[k][2].append(f"\t\t\t\\textbf{{{mtxname.replace('_','\\_')}}} & {min_t * 1000:.3f} & {max_t * 1000:.3f} & {avg_t * 1000:.3f} & {var_t:.5f} & {gflops:.4f} \\\\")

    for ser in ser_data_aggregate:
        mtxname = ser[:find_underscore_from_end(ser, 2)]
        min_t = min(ser_data_aggregate[ser])
        max_t = max(ser_data_aggregate[ser])
        avg_t = avg(ser_data_aggregate[ser])
        var_t = var(ser_data_aggregate[ser])
        gflops = (2 * mtxs_with_nonzeroes[mtxname] / avg_t) / 1e9
        if f"{ser}_ser".endswith(k):
            diz[k][2].append(f"\t\t\t\\textbf{{{mtxname.replace('_','\\_')}}} & {min_t * 1000:.3f} & {max_t * 1000:.3f} & {avg_t * 1000:.3f} & {var_t:.5f} & {gflops:.4f} \\\\")
    
    for mt in mt_data_aggregate:
        mtxname = mt[:find_underscore_from_end(mt, 3)]
        min_t = min(mt_data_aggregate[mt])
        max_t = max(mt_data_aggregate[mt])
        avg_t = avg(mt_data_aggregate[mt])
        var_t = var(mt_data_aggregate[mt])
        gflops = (2 * mtxs_with_nonzeroes[mtxname] / avg_t) / 1e9
        if f"{mt}_mt".endswith(k):
            diz[k][2].append(f"\t\t\t\\textbf{{{mtxname.replace('_','\\_')}}} & {min_t * 1000:.3f} & {max_t * 1000:.3f} & {avg_t * 1000:.3f} & {var_t:.5f} & {gflops:.4f} \\\\")

print(f"""\t\\subsection{{Nonzeri delle matrici}}
\t\t\\begin{{table}}[H]
\t\t\\centering
\t\t\\begin{{tabular}}{{| l | c |}}
\t\t\t\\hline
\t\t\t\\textbf{{Matrix}} & \\textbf{{Num. di nonzero}} \\\\
\t\t\t\\hline
\t\t\t\\hline
{'\n'.join(mtxnzs_latex)}
\t\t\t\\hline
\t\t\\end{{tabular}}
\t\t\\caption{{Numero di nonzeroes per le matrici}}
\t\t\\label{{table:mtxnzs}}
\t\\end{{table}}
\t\\pagebreak
""")

for k in diz.keys():
    print(f"""\t\\subsection{{{diz[k][0]}}}
\t\\begin{{table}}[H]
\t\t\\centering
\t\t\\begin{{tabular}}{{| l | c c c c | c |}}
\t\t\t\\hline
\t\t\t\\textbf{{Matrix}} & \\textbf{{Min($T$)$[ms]$}} & \\textbf{{Max($T$)$[ms]$}} & 
\t\t\t\\textbf{{Avg($T$)$[ms]$}} & \\textbf{{Var($T$)}} & \\textbf{{GFLOPS}} \\\\ %%% header %%%
\t\t\t\\hline
\t\t\t\\hline
{'\n'.join(diz[k][2])}
\t\t\t\\hline
\t\t\\end{{tabular}}
\t\t\\caption{{{diz[k][1]}}}
\t\t\\label{{table:{k}}}
\t\\end{{table}}
\t\\pagebreak
""")