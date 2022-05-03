with source_data as (
     select * from {{ ref('raw_sample_batch') }}
)

select * from source_data
