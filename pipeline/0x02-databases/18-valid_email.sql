-- creates a trigger before update
DELIMITER $$

CREATE  TRIGGER  triggerr_name
    BEFORE UPDATE
    ON users FOR EACH ROW
BEGIN
    UPDATE users
    SET valid_email = 0
    WHERE users.email = NEW.users.email;
    
END $$

DELIMITER ;