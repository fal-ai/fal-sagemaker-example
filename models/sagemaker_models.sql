{{ config(materialized='ephemeral') }}

SELECT * FROM {{ target.schema }}.{{ model.name }}
