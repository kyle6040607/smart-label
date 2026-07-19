-- 切換桶子
USE smart_label_db;

-- User
CREATE TABLE if not exists user (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    account VARCHAR(64) NOT NULL, 
    pw_hash VARCHAR(255) NOT NULL,
    line_id VARCHAR(64) NOT NULL,
    display_name VARCHAR(64),
    create_at datetime DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_account UNIQUE (account),
    CONSTRAINT uq_line_id UNIQUE (line_id)
)ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- project
CREATE TABLE if not exists project (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    owner_id INT UNSIGNED NOT NULL,
    name VARCHAR(128) NOT NULL,
    mode ENUM('novice', 'expert') DEFAULT 'novice',
    create_at datetime DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_project_owner_id_user_id FOREIGN KEY (owner_id) REFERENCES user(id) ON DELETE CASCADE
)ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- photo
CREATE TABLE if not exists photo(
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    project_id INT UNSIGNED NOT NULL,
    path VARCHAR(512) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    width INT UNSIGNED NOT NULL,
    height INT UNSIGNED NOT NULL,
    create_at datetime DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_photo_project_id_project_id FOREIGN KEY (project_id) REFERENCES project(id) ON DELETE CASCADE
)ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- segment
CREATE TABLE if not exists segment(
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    photo_id INT UNSIGNED NOT NULL,
    mask_path VARCHAR(512) NOT NULL,
    bbox JSON NOT NULL,
    area INT UNSIGNED NOT NULL,
    predicted_label VARCHAR(64) NOT NULL,
    confidence FLOAT UNSIGNED DEFAULT 0,
    needs_review BOOLEAN DEFAULT false,
    human_label VARCHAR(64) NOT NULL,
    reviewed BOOLEAN DEFAULT false,
    CONSTRAINT fk_segment_photo_id_photo_id FOREIGN KEY (photo_id) REFERENCES photo(id) ON DELETE CASCADE
)ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- example
CREATE TABLE if not exists example(
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    project_id INT UNSIGNED NOT NULL,
    label VARCHAR(64) NOT NULL,
    feature JSON NOT NULL,
    source_segment_id INT UNSIGNED NOT NULL,
    create_at datetime DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_example_project_id_project_id FOREIGN KEY (project_id) REFERENCES project(id) ON DELETE CASCADE,
    CONSTRAINT fk_example_source_segment_id_segment_id FOREIGN KEY (source_segment_id) REFERENCES segment(id) ON DELETE CASCADE
)ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- history
CREATE TABLE if not exists history(
    id INT UNSIGNED AUTO_INCREMENT PRIMARY Key,
    user_id INT UNSIGNED NOT NULL,
    project_id INT UNSIGNED NOT NULL,
    action VARCHAR(64) NOT NULL,
    create_at datetime DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_history_user_id_user_id FOREIGN KEY (user_id) REFERENCES user(id) ON DELETE CASCADE,
    CONSTRAINT fk_history_project_id_project_id FOREIGN KEY (project_id) REFERENCES project(id) ON DELETE CASCADE
)ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;