CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(500) NOT NULL,
    description TEXT,
    category VARCHAR(200),
    price DECIMAL(10, 2),
    stock INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    order_id VARCHAR(100) NOT NULL,
    status VARCHAR(100),
    customer_name VARCHAR(200),
    total_amount DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    email VARCHAR(200),
    phone VARCHAR(50),
    address TEXT
);

INSERT INTO products (name, description, category, price, stock) VALUES
('iPhone 15 Pro', '苹果最新旗舰手机，A17芯片，6.7英寸超视网膜XDR显示屏，支持5G网络', '手机', 7999.00, 100),
('MacBook Pro 14英寸', 'M3芯片，14英寸Liquid Retina XDR显示屏，16GB内存，512GB固态硬盘', '电脑', 14999.00, 50),
('AirPods Pro 2', '主动降噪，空间音频，通透模式，长达6小时续航', '配件', 1899.00, 200),
('iPad Air 5', '10.9英寸Liquid Retina显示屏，M1芯片，支持Apple Pencil', '平板', 4799.00, 80),
('Apple Watch Series 9', '更亮的显示屏，更快的芯片，全新的双指互点手势', '手表', 2999.00, 150);

INSERT INTO orders (order_id, status, customer_name, total_amount) VALUES
('ORD202401001', '已完成', '张三', 7999.00),
('ORD202401002', '处理中', '李四', 14999.00),
('ORD202401003', '已取消', '王五', 1899.00),
('ORD202401004', '已发货', '赵六', 4799.00),
('ORD202401005', '已完成', '孙七', 2999.00);

INSERT INTO customers (name, email, phone, address) VALUES
('张三', 'zhangsan@example.com', '13800138001', '北京市朝阳区建国路88号'),
('李四', 'lisi@example.com', '13900139002', '上海市浦东新区世纪大道100号'),
('王五', 'wangwu@example.com', '13700137003', '广州市天河区珠江新城A栋'),
('赵六', 'zhaoliu@example.com', '13600136004', '深圳市南山区科技园路50号'),
('孙七', 'sunqi@example.com', '13500135005', '杭州市西湖区文三路666号');
