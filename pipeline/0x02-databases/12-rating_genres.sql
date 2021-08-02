-- lists all genres in the database hbtn_0d_tvshows_rate by their rating

-- lists all genres in the database hbtn_0d_tvshows_rate by their rating

SELECT tvg.name, SUM(tvsr.rate) AS rating
FROM tv_genres tvg JOIN tv_show_genres tvsg ON tvsg.genre_id = tvg.id JOIN tv_show_ratings tvsr ON tvsr.show_id = tvsg.show_id
GROUP BY tvg.name
ORDER BY rating DESC;