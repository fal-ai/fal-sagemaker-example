with source_data as (
     select * from {{ ref('raw_census_data') }}
)

select * from source_data
