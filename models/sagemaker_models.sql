{{ config(materialized='ephemeral') }}
-- {{ ref('training_sample') }}

SELECT * FROM {{ target.schema }}.{{ model.name }}
