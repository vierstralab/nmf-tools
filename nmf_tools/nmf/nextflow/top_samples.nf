#!/usr/bin/env nextflow
nextflow.enable.dsl = 2


process find_top_samples {

    conda params.conda
    tag "${prefix}"
    publishDir "${params.outdir}/nmf/${prefix}", pattern: "${name}"
    label "highmem"

    input:
        tuple val(prefix), path(W)

    output:
        tuple val(prefix), path("*.*.component_${prefix}.bw"), path(name)

    script:
    name = "${prefix}.top_samples.tsv"
    """
    python3 $moduleDir/bin/find_top_samples.py \
        ${prefix} \
        ${params.nmf_config} \
        ${W} \
        --top ${params.top_count}
    """
}


process top_samples_track {

    scratch true
    conda params.conda
    tag "${prefix}:${component}"
    publishDir "${params.outdir}/nmf/${prefix}/top_samples"

    input:
        tuple val(prefix), val(component), path(density_bw, stageAs: "?/*")
    
    output:
        tuple val(prefix), val(component), path(name)
    
    script:
    name = "${prefix}.${component}.top_samples.bw"
    bg = "${prefix}.${component}.top_samples.bg"
    """
    wiggletools write_bg ${bg} mean ${density_bw}
    bedGraphToBigWig "${bg}" "${params.chrom_sizes}" "${name}"
    """
}

workflow findTop {
    take:
        data // 
    main:
        top_samples = data
            | map(it -> tuple(it[0], it[1])) // prefix, W
            | find_top_samples
            | map(it -> tuple(it[0], it[1]))
            | transpose() // prefix, bw_file
            | map(it -> tuple(it[0], it[1].simpleName, it[1]))
            | groupTuple(by: [0, 1]) // component, prefix, bw_files
            | top_samples_track
            | collectFile(
                skip: 1,
                keepHeader: true,
                storeDir: "${params.outdir}/nmf/"
            ) {
                [
                    "${it[0]}/${it[0]}.component_tracks_meta.tsv", //name
                    "component\taggregated_bw\n${it[1]}\t${params.outdir}/nmf/${it[0]}/top_samples/${it[2].name}\n" // content
                ]
            }

    emit:
        top_samples
}