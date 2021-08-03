--  script that creates a stored procedure ComputeAverageWeightedScoreForUser that computes and store
-- the average weighted score for a student.
DROP PROCEDURE ComputeAverageWeightedScoreForUser;
DELIMITER $$
CREATE  PROCEDURE ComputeAverageWeightedScoreForUser(
	IN user_idd INTEGER)
	BEGIN
		UPDATE users SET average_score=(
			SELECT SUM(co.score*pr.weight)/SUM(pr.weight) FROM corrections co
				JOIN projects  pr ON co.project_id=pr.id
				WHERE co.user_id=user_idd) WHERE id=user_idd;
	END $$
DELIMITER ;
