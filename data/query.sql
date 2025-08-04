WITH tb_fl_churn AS (

    SELECT t1.dtRef,
        t1.idCustomer,
        CASE WHEN t2.idCustomer IS NULL THEN 1 ELSE 0 END AS flChurn

    FROM fs_general AS t1

    LEFT JOIN fs_general AS t2
    ON t1.idCustomer = t2.idCustomer
    AND t1.dtRef = date(t2.dtRef, '-21 day')

    WHERE (t1.dtRef < DATE('2024-06-20', '-21 day')
    AND strftime('%d', t1.dtRef) = '01')
    OR t1.dtRef = DATE('2024-06-20', '-21 day')

    order by 1,2

)

SELECT t1.*
FROM tb_fl_churn AS t1