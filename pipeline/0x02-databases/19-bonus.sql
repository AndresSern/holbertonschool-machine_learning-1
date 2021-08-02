--  stored procedure AddBonus that adds a new correction for a student.
DELIMITER $$

CREATE  PROCEDURE AddBonus (
	IN user_id INT,
    IN project_name varchar(255),
    IN score INT

)
BEGIN

IF NOT EXISTS(SELECT id FROM projects WHERE name = project_name) THEN
    INSERT INTO projects (name) VALUES (project_name);
END IF;
SELECT id INTO @ide FROM projects WHERE projects.name = project_name;
INSERT INTO corrections (user_id, project_id, score) VALUES (user_id, @ide, score);
END$$

DELIMITER ;
