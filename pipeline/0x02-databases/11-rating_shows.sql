-- lists all shows from hbtn_0d_tvshows_rate by their rating

SELECT tvs.title, SUM(tvsr.rate) AS rating
FROM tv_shows tvs JOIN tv_show_ratings tvsr ON tvsr.show_id = tvs.id
GROUP by tvs.title
ORDER BY rating DESC;
