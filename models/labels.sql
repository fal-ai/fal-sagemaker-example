with source_data as (
     select * from {{ ref('raw_labels') }}
)

select * from source_data
