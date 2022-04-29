with source_data as (
     select * from {{ ref('raw_features') }}
)

select * from source_data
