-- lists all shows contained in hbtn_0d_tvshows without a genre linked.
SELECT sh.title, ge.genre_id FROM tv_shows sh LEFT JOIN tv_show_genres  ge ON sh.id = ge.show_id WHERE ge.show_id IS NULL ORDER BY sh.title, ge.genre_id;
