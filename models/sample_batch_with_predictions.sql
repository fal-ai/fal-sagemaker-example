{{ config(materialized='ephemeral') }}
-- {{ ref('sample_batch') }}

SELECT * FROM {{ target.schema }}.{{ model.name }}
