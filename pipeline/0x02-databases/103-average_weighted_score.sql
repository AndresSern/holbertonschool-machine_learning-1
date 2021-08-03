-- DROP PROCEDURE ComputeAverageWeightedScoreForUser;
DELIMITER $$
CREATE  PROCEDURE ComputeAverageWeightedScoreForUser(
	IN user_id INT)
	BEGIN
		UPDATE users SET average_score=(
			SELECT SUM(co.score*pr.weight)/SUM(pr.weight) FROM corrections co
				JOIN projects  pr ON co.project_id=pr.id
				WHERE co.user_id=user_id) WHERE id=user_id;
	END $$
DELIMITER ;
