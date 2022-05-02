with source_data as (
     select * from {{ ref('raw_population') }}
)

select * from source_data
