select * from logs order by log_dttm desc;


select distinct s.* from stations s
inner join arrivals a on a.station_nm = s.station_nm
WHERE ST_Within(
    s.loc,
    ST_MakeEnvelope(155, 48, 167, 62, 4326)   
)


select * from events 
WHERE ST_Within(
    loc,
    ST_MakeEnvelope(156, 51, 163, 62, 4326)
   
);


select * from arrivals
where event_id = 15813965;


select event_id from events
group by event_id
having count(event_id) >= 2;


select * from arrivals
where arrival_dttm::date = '2013-05-08'


select 
	e.event_id,
	e.loc as event_loc,
	e.event_dttm as event_dttm, 
	s.loc as station_loc,
	a.arrival_dttm as arrival_dttm,
	a.arrival_type,
	a.arrival_dttm - e.event_dttm as delta_t
from events e
inner join arrivals a on
	a.event_id  = e.event_id
inner join stations s on
	s.station_nm = a.station_nm and s.network_nm = a.network_nm
WHERE 
	ST_Within(
    	e.loc,
    	ST_MakeEnvelope(156, 51, 163, 62, 4326)
	)
	and
	ST_Within(
    	s.loc,
    	ST_MakeEnvelope(156, 51, 163, 62, 4326)
	)
	

-- select arrival_type, count(*) from t group by arrival_type 

	
SELECT
    e.event_id,
    s.station_nm,
    s.network_nm,
    MIN(e.event_dttm) AS event_dttm,
    -- Агрегируем по типам волн
    MAX(CASE 
        WHEN a.arrival_type LIKE 'P%' THEN a.arrival_dttm - e.event_dttm 
    END) AS delta_t_p,
    MAX(CASE 
        WHEN a.arrival_type LIKE 'S%' THEN a.arrival_dttm - e.event_dttm 
    END) AS delta_t_s,
    -- Определяем wave_subtype
    CASE
        WHEN a.arrival_type LIKE '%g' THEN 'g'
        WHEN a.arrival_type LIKE '%n' THEN 'n'
        WHEN a.arrival_type LIKE '%b' THEN 'b'
        ELSE NULL
    END AS wave_subtype
FROM events e
JOIN arrivals a ON a.event_id = e.event_id
JOIN stations s ON s.station_nm = a.station_nm AND s.network_nm = a.network_nm
WHERE 
    ST_Within(e.loc, ST_MakeEnvelope(156, 51, 163, 62, 4326))
    AND ST_Within(s.loc, ST_MakeEnvelope(156, 51, 163, 62, 4326))
    AND a.arrival_type IN ('P', 'S', 'Pg', 'Sg', 'Pn', 'Sn', 'Pb', 'Sb')
GROUP BY
    e.event_id,
    s.station_nm,
    s.network_nm,
    wave_subtype;


SELECT
    e.event_id,
    e.loc,
    s.station_nm,
    s.network_nm,
    MIN(e.event_dttm) AS event_dttm,
    -- Агрегируем по типам волн
    MAX(CASE 
        WHEN a.arrival_type LIKE 'P%' THEN a.arrival_dttm - e.event_dttm 
    END) AS delta_t_p,
    MAX(CASE 
        WHEN a.arrival_type LIKE 'S%' THEN a.arrival_dttm - e.event_dttm 
    END) AS delta_t_s,
    -- Определяем wave_subtype
    CASE
        WHEN a.arrival_type LIKE '%g' THEN 'g'
        WHEN a.arrival_type LIKE '%n' THEN 'n'
        WHEN a.arrival_type LIKE '%b' THEN 'b'
        ELSE NULL
    END AS wave_subtype
    FROM events e
    JOIN arrivals a ON a.event_id = e.event_id
    JOIN stations s ON s.station_nm = a.station_nm AND s.network_nm = a.network_nm
    WHERE 
    ST_Within(e.loc, ST_MakeEnvelope(-115, 32, -120, 35, 4326))
    AND ST_Within(s.loc, ST_MakeEnvelope(-115, 35, -120, 32, 4326))
    AND a.arrival_type IN ('P', 'S', 'Pg', 'Sg', 'Pn', 'Sn', 'Pb', 'Sb')
    GROUP BY
    e.event_id,
    s.station_nm,
    s.network_nm,
    wave_subtype; 


select count(e.event_id), e.event_dttm::date
from events e
group by 2
order by 2

select * from events e
where ST_Within(e.loc, ST_MakeEnvelope(156, 49, 166, 62, 4326))

--e.event_dttm::date < '2017-03-06' and e.event_dttm::date > '2016-03-01'



--select e.loc, s.loc from

with dat as(
select distinct
    e.loc as e_loc,
    s.loc as s_loc
    FROM events e
    JOIN arrivals a ON a.event_id = e.event_id
    JOIN stations s ON s.station_nm = a.station_nm AND s.network_nm = a.network_nm
    WHERE 
    ST_Within(e.loc, ST_MakeEnvelope(156, 51, 163, 62, 4326))
    AND ST_Within(s.loc, ST_MakeEnvelope(156, 51, 163, 62, 4326))
   -- AND a.arrival_type IN ('P', 'S', 'Pg', 'Sg', 'Pn', 'Sn', 'Pb', 'Sb')
)
select * from (
	select e_loc, row_number() over () as rn from (select distinct e_loc from dat)) t1
	full join (
	select s_loc, row_number() over () as rn from (select distinct s_loc from dat)) t2 
	using(rn)
order by rn


select * from events limit 10


-- CREATE TABLE travel_times (
--    experiment_nm TEXT NOT NULL,
--    event_id TEXT NOT NULL,
--    station_nm TEXT NOT NULL,
--    network_nm TEXT,
--    delta_t_p FLOAT,       -- длительность P-волны
--    delta_t_s FLOAT,       -- длительность S-волны
--    wave_subtype TEXT
--);

-- drop table travel_times

--CREATE TABLE experiments (
--    experiment_id TEXT,   -- уникальный идентификатор эксперимента
--    loc GEOGRAPHY(POINT, 4326), 
--    r_stations INTEGER,                   -- количество станций
--   r_events INTEGER,                     -- количество событий
--    from_dttm TIMESTAMP, -- время начала
--    to_dttm TIMESTAMP,   -- время конца
--  vp_vs_params JSONB                     -- параметры vp/vs в формате JSON
--);




select * from experiments
where experiment_nm = '20251202214050_exp_2007-01-01_2022-12-31_48_155'
limit 10




select 
distinct e.event_id, s.station_nm, s.network_nm, e.loc as event_loc, s.loc as station_loc 
--count(distinct e.event_id || )
from travel_times tt
inner join events e on e.event_id = tt.event_id::int
inner join stations s on s.station_nm = tt.station_nm and s.network_nm = tt.network_nm
where experiment_nm = '20251202214050_exp_2007-01-01_2022-12-31_48_155'
and e.event_dttm::date > '2010-01-01'
and e.event_dttm::date < '2010-01-31';

select distinct e.event_id, s.station_nm, s.network_nm, e.loc as event_loc, s.loc as station_loc from arrivals a
inner join events e on e.event_id = a.event_id
inner join stations s on s.station_nm = a.station_nm and s.network_nm = a.network_nm
where truegis
and ST_Within(e.loc, ST_MakeEnvelope(155, 48, 167, 60, 4326))
and ST_Within(s.loc, ST_MakeEnvelope(155, 48, 167, 60, 4326))
and e.event_dttm::date > '2010-01-01'
and e.event_dttm::date < '2010-01-31';

select e.event_id, e.loc as event_loc from events e
where true
and ST_Within(e.loc, ST_MakeEnvelope(155, 48, 167, 60, 4326))
and e.event_dttm::date > '2010-01-01'
and e.event_dttm::date < '2010-12-31';





select * from events e 
where true
--and e.event_type_certainty ='known'
and e.event_dttm::date > '2000-01-01'
and e.event_dttm::date < '2022-12-31'


select 
e.geometry as loc,
(e.params::jsonb->>'n_points')::numeric as n_points,
((e.params::jsonb->>'regression_free')::jsonb->>'k')::numeric as k1,
((e.params::jsonb->>'regression_free')::jsonb->>'r2')::numeric as misfit_k1,
((e.params::jsonb->>'regression_zero_intercept')::jsonb->>'k')::numeric as k2,
((e.params::jsonb->>'regression_zero_intercept')::jsonb->>'r2')::numeric as misfit_k2
from experiments e
--where e.experiment_nm = '20251126113633_exp_2005-01-01_2009-12-31_48_155' and (e.params::jsonb->>'n_points')::numeric > 5500;

select 
e.t_from,
e.t_to,
AVG((e.params::jsonb->>'n_points')::numeric) as n_points
from experiments e
where experiment_nm = '20251202165343_exp_2001-01-01_2020-12-31_48_155'
group by 1,2



select distinct event_id, loc, event_type_certainty from events
where event_dttm >= '2010-01-01' and event_dttm <= '2010-12-31' and ST_Within(loc, ST_MakeEnvelope(155, 48, 167, 60, 4326)) 

select count(distinct a.pick_id ), date_trunc('year', e.event_dttm::date)::date as year_, e.event_type_certainty  from events e
inner join arrivals a on a.event_id = e.event_id 
where ST_Within(e.loc, ST_MakeEnvelope(155, 48, 167, 60, 4326))
group by 2,3

select count(distinct e.event_id), date_trunc('year', e.event_dttm::date)::date as year_, e.event_type_certainty  from events e
where ST_Within(e.loc, ST_MakeEnvelope(155, 48, 167, 60, 4326))
group by 2,3

select count(distinct e.event_id ), date_trunc('year', e.event_dttm::date)::date as year_, e.event_type_certainty  from events e
inner join magnitudes m
on m.event_id = e.event_id and m.mag_value >= 1.0
where ST_Within(e.loc, ST_MakeEnvelope(155, 48, 167, 60, 4326))
group by 2,3


select distinct e.loc, e.event_dttm::date from events e
inner join arrivals a
on a.event_id = e.event_id
inner join magnitudes m
on m.event_id = e.event_id and m.mag_value >= 7.0
where
e.event_type_certainty = 'known'
and e.event_dttm >= '2007-01-01' and e.event_dttm <= '2022-12-31' and ST_Within(e.loc, ST_MakeEnvelope(155, 48, 167, 60, 4326))


select log_nm, log_dttm, load_status, (params::jsonb->>'depth')::numeric as depth, params from logs
where log_nm = 'load_arrivals' 
order by log_dttm desc;

select log_nm, log_dttm, load_status, (params::jsonb->>'depth')::numeric as depth, params from logs
where log_nm = 'load_events_mags' 
order by log_dttm desc;



select schemaname as table_schema,
    relname as table_name,
    pg_size_pretty(pg_total_relation_size(relid)) as total_size,
    pg_size_pretty(pg_relation_size(relid)) as data_size,
    pg_relation_size(relid) as data_size,
    pg_size_pretty(pg_total_relation_size(relid) - pg_relation_size(relid))
      as external_size,
      pg_relation_size(relid)
from pg_catalog.pg_statio_user_tables
order by pg_total_relation_size(relid) desc
,         pg_relation_size(relid) desc


select
		MAX(CASE WHEN a.arrival_type LIKE 'P%' THEN a.pick_id END) AS p_pick_id,
        MAX(CASE WHEN a.arrival_type LIKE 'S%' THEN a.pick_id END) AS s_pick_id,
        e.event_id,
        s.station_nm,
        s.network_nm,
        MIN(e.event_dttm) AS event_dttm,
        -- Time deltas
        MAX(CASE WHEN a.arrival_type LIKE 'P%' THEN a.arrival_dttm - e.event_dttm END) AS delta_t_p,
        MAX(CASE WHEN a.arrival_type LIKE 'S%' THEN a.arrival_dttm - e.event_dttm END) AS delta_t_s,
        -- Wave subtype
        CASE
            WHEN a.arrival_type LIKE '%g' THEN 'g'
            WHEN a.arrival_type LIKE '%n' THEN 'n'
            WHEN a.arrival_type LIKE '%b' THEN 'b'
            ELSE NULL
        END AS wave_subtype
    FROM events e
    JOIN arrivals a ON a.event_id = e.event_id
    JOIN stations s ON s.station_nm = a.station_nm 
                    AND s.network_nm = a.network_nm
    WHERE 
        -- Events radius filter
        ST_DWithin(
            e.loc,
            ST_SetSRID(ST_MakePoint(157, 50), 4326)::geography,
            400000
        )
        -- Stations radius filter
        AND ST_DWithin(
            s.loc,
            ST_SetSRID(ST_MakePoint(157, 50), 4326)::geography,
            550000
        )
        AND a.arrival_type IN ('P','S','Pg','Sg','Pn','Sn','Pb','Sb')
        AND e.event_dttm::date >= '2010-01-01'
        AND e.event_dttm::date <= '2010-12-31'
        -- AND e.event_type_certainty = 'known'
    GROUP BY
        e.event_id,
        s.station_nm,
        s.network_nm,
        wave_subtype;



--ALTER TABLE travel_times 
--ALTER COLUMN p_pick_id type INT8,
--ALTER COLUMN s_pick_id type INT8;

