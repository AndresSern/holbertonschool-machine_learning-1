-- lists all shows contained in hbtn_0d_tvshows that have at least one genre linked.

-- Each record should display: tv_shows.title - tv_show_genres.genre_id
-- Results must be sorted in ascending order by tv_shows.title and tv_show_genres.genre_id
SELECT sh.title, ge.genre_id FROM tv_shows sh
JOIN tv_show_genres ge ON sh.id = ge.show_id ORDER BY sh.title, ge.genre_id;

