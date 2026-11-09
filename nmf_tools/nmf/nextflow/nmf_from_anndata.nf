#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { findTop } from "./top_samples"

process fit_nmf {
	tag "${prefix}"
	conda params.conda
    publishDir "${params.outdir}/nmf/${prefix}", pattern: "${prefix}.*"
    label "highmem"

	input:
		val prefix

	output:
        tuple val(prefix), path("${prefix}.W.npy"), path("${prefix}.H.npy"), path("${prefix}.non_zero_peaks_mask.txt"), emit: results
        path("${prefix}.samples_mask.txt"), emit: samples_mask

	script:
	"""
    python3 $moduleDir/bin/run_NMF.py \
        ${prefix} \
        ${params.nmf_config}
	"""
}

process visualize_nmf {
	tag "${prefix}"
	conda params.conda
    publishDir "${params.outdir}/nmf/${prefix}"
    label "highmem"
    errorStrategy 'ignore'

	input:
        tuple val(prefix), path(W), path(H), path(non_zero_peaks_mask)

	output:
        tuple val(prefix), path("*.pdf")

	script:
	"""
    python3 $moduleDir/bin/visualize_nmf.py \
        ${prefix} \
        ${params.nmf_config} \
        ${W} \
        ${H} \
        ${non_zero_peaks_mask}
	"""
}


// nextflow run ~/packages/nmf_tools/nmf/nextflow/nmf_from_anndata.nf -profile Altius -resume
workflow {
    prefixes = Channel.fromPath(params.nmf_params)
        | splitCsv(header: false)
        | map(it -> it[0])
        | distinct { it }
    fit_nmf(prefixes)
    fit_nmf.out.results | (visualize_nmf & findTop)
}


// Entry for visuzizations only
workflow visualize {
    params.nmf_results_path = "${params.outdir}"  // default location where nf output goes
    println "Visualizing NMF results from params.nmf_params = ${params.nmf_params}. Assuming nf output folder to be params.nmf_results_path=${params.nmf_results_path}"
    
    Channel.fromPath(params.nmf_params)
        | splitCsv(header: false)
        | map(it -> it[0])
        | distinct { it }
        | map( 
            it -> tuple(
                it,
                file("${params.nmf_results_path}/nmf/${it}/${it}.W.npy"),
                file("${params.nmf_results_path}/nmf/${it}/${it}.H.npy"),
                file("${params.nmf_results_path}/nmf/${it}/${it}.non_zero_peaks_mask.txt"),
            )
        )
        | visualize_nmf
}

// Defunc
workflow topSamples {
    params.nmf_results_path = "${params.outdir}"  // default location where nf output goes
    println "Visualizing NMF results from params.nmf_params = ${params.nmf_params}. Assuming nf output folder to be params.nmf_results_path=${params.nmf_results_path}"
    
    Channel.fromPath(params.nmf_params)
        | splitCsv(header: false)
        | map(it -> it[0])
        | distinct { it }
        | map( 
            it -> tuple(
                it,
                file("${params.nmf_results_path}/nmf/${it}/${it}.W.npy"),
                file("${params.nmf_results_path}/nmf/${it}/${it}.H.npy"),
                file("${params.nmf_results_path}/nmf/${it}/${it}.non_zero_peaks_mask.txt"),
            )
        )
        | findTop
}


// process add_metadata {
//     conda params.conda
//     publishDir "${params.outdir}"
//     errorStrategy 'ignore'

//     output:
//         path name

//     script:
//     name = "nmf_meta+matrices.tsv"
//     """
//     python3 $moduleDir/bin/nmf/add_metadata.py \
//         ${params.nmf_config} \
//         ${params.outdir}/nmf \
//         ${name}
//     """
// }