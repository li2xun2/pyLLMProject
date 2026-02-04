import jwt

# 使用与Java后端相同的密钥
SECRET_KEY = "abcdefghijkomnopqrstuvwxyx"
ALGORITHM = "HS512"

# 用户的token
user_token = "eyJhbGciOiJIUzUxMiJ9.eyJsb2dpbl9tZW1iZXJfa2V5IjoiNTc3NmU0OTgtMGZhNy00ZTE4LTk5M2EtMGZiOTZlNWY0YTY2In0.qR2kOkcsf9OaQpPOvpFvniDUHFYPQEMfgDh3YN_sL0Z2M8nkEHP2EoOng23i7ehzDpGQiWnmWoS9unA0V7np6g"

print("=== Token解析结果 ===")
print(f"Token: {user_token}")
print()

try:
    # 解析token
    payload = jwt.decode(user_token, SECRET_KEY, algorithms=[ALGORITHM])
    print("✅ Token解析成功！")
    print(f"Payload内容: {payload}")
    print()
    print("=== 详细信息 ===")
    for key, value in payload.items():
        print(f"{key}: {value}")
    print()
    print("=== 分析 ===")
    print("1. token中包含的字段:")
    for key in payload.keys():
        print(f"   - {key}")
    print()
    print("2. 是否包含用户ID:")
    user_id_fields = ['sub', 'userId', 'id', 'login_member_key', 'member_id', 'user_id']
    has_user_id = False
    for field in user_id_fields:
        if field in payload:
            print(f"   ✅ 包含字段 '{field}': {payload[field]}")
            has_user_id = True
    if not has_user_id:
        print("   ❌ 未找到用户ID字段")
    print()
    print("3. 可能的用户标识:")
    print(f"   login_member_key: {payload.get('login_member_key', 'N/A')}")
    print()
    print("4. 结论:")
    print("   - token中存储的是登录标识信息")
    print("   - 具体来说是login_member_key字段，值为: 5776e498-0fa7-4e18-993a-0fb96e5f4a66")
    print("   - 这个值可以作为用户的唯一标识")
except jwt.InvalidTokenError as e:
    print(f"❌ Token解析失败: {str(e)}")
    print("可能的原因:")
    print("1. 密钥不正确")
    print("2. token已过期")
    print("3. token被篡改")
except Exception as e:
    print(f"❌ 解析出错: {type(e).__name__}: {str(e)}")

# 尝试不验证签名来查看token内容
print()
print("=== 不验证签名解析 ===")
try:
    payload = jwt.decode(user_token, options={"verify_signature": False})
    print("✅ 不验证签名解析成功！")
    print(f"Payload内容: {payload}")
    print()
    print("Token头部信息:")
    import base64
    import json
    # 解析token头部
    header_part = user_token.split('.')[0]
    # 修复base64填充
    padding = '=' * ((4 - len(header_part) % 4) % 4)
    header_json = base64.urlsafe_b64decode(header_part + padding)
    header = json.loads(header_json)
    print(f"算法: {header.get('alg')}")
    print(f"类型: {header.get('typ')}")
except Exception as e:
    print(f"❌ 解析失败: {type(e).__name__}: {str(e)}")
