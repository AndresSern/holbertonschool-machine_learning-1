-- lists all genres from hbtn_0d_tvshows and displays the number of shows linked to each

SELECT tvg.name AS genre,  COUNT(tvs.genre_id) AS number_of_shows
FROM tv_show_genres tvs JOIN tv_genres tvg ON tvs.genre_id = tvg.id
GROUP by tvg.name
ORDER BY number_of_shows DESC;
