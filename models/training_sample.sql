with source_data as (
     select * from {{ ref('raw_training_data') }}
)

select * from source_data
