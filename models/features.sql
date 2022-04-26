select
        age,
        workclass,
        "education.num" as education_num,
        "marital.status" as marital_status,
        occupation,
        relationship,
        race,
        sex,
        "capital.gain" as capital_gain,
        "capital.loss" as capital_loss,
        "hours.per.week" as hours_per_week,
        "native.country" as native_country
from {{ ref("census_base" )}}
