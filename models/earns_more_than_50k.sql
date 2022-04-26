select case when income='<=50K' then 0
            else 1 end
            as above
from {{ ref('census_base') }}

