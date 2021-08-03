--  script that creates a stored procedure ComputeAverageWeightedScoreForUser that computes and store
-- the average weighted score for a student.
-- DROP PROCEDURE ComputeAverageWeightedScoreForUser;
DELIMITER $$
CREATE  PROCEDURE ComputeAverageWeightedScoreForUser(
	IN user_id INT)
	BEGIN
		UPDATE users SET average_score=(
			SELECT SUM(score*weight)/SUM(weight) FROM corrections co
				JOIN projects  pr ON co.project_id=pr.id
				WHERE pr.user_id=user_id) WHERE id=user_id;
	END $$
DELIMITER ;
