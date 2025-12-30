#include <vector>
#include <iostream>
#include <random>
#include <string>
#include <memory>
#include <algorithm>
#include <functional>

//For MAC
#if defined(__APPLE__)
    #define GL_SILENCE_DEPRECATION 
    #define GLFW_INCLUDE_NONE
    #include <OpenGL/gl3.h>
#else
    #include <glad/glad.h>
    #include <GLFW/glfw3.h>
#endif


#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

//Glm
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"


// System state
const uint16_t SCREEN_WIDTH = 1280;
const uint16_t SCREEN_HEIGHT = 720;

namespace {
    //imgui
    float worldSize = 10.0f; 
    bool isPaused = false;
    float timeScale = 1.0f;
    bool useOctree = true;
    bool showOctreeDebug = true;
    int64_t countCollisions = 0;

    bool isMenuVisible = true;
    int sizeObject = 1; //0=small, 1=med, 2=big, imgui requiere int

}//Edit;

// Delta Time
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// Camera
glm::vec3 cameraPos = glm::vec3(0.0f, 10.0f, 30.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp  = glm::vec3(0.0f, 1.0f, 0.0f);
float yaw   = -90.0f; //Horizontal rotation
float pitch = -20.0f; //Veritcal rotation
float lastX = SCREEN_WIDTH / 2.0f;
float lastY = SCREEN_HEIGHT / 2.0f;
bool firstMouse = true;

// Shaders
const char* vertexShaderSource = R"(
#version 410 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
out vec3 Normal;
out vec3 FragPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;  
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
)";

const char* fragmentShaderSource = R"(
#version 410 core
out vec4 FragColor;
in vec3 Normal;
in vec3 FragPos;
uniform vec3 objectColor;
uniform vec3 lightPos;
uniform vec3 viewPos;
void main() {
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * vec3(1.0);
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * vec3(1.0);
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * vec3(1.0);  
    vec3 result = (ambient + diffuse + specular) * objectColor;
    FragColor = vec4(result, 1.0);
}
)";

// random
std::random_device rd;
std::mt19937 gen(rd());
float randomFloat(float min, float max) {
    std::uniform_real_distribution<float> dis(min, max);
    return dis(gen);
}

// class shader to compile shader
class Shader {
public:
    unsigned int ID;
    Shader(const char* vCode, const char* fCode) {
        unsigned int vertex, fragment;
        int success; char infoLog[512];
        vertex = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex, 1, &vCode, NULL);
        glCompileShader(vertex);
        fragment = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment, 1, &fCode, NULL);
        glCompileShader(fragment);
        ID = glCreateProgram();
        glAttachShader(ID, vertex);
        glAttachShader(ID, fragment);
        glLinkProgram(ID);
        glDeleteShader(vertex);
        glDeleteShader(fragment);
    }
    void use() { glUseProgram(ID); }
    void setMat4(const std::string &name, const glm::mat4 &mat) const {
        glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
    }
    void setVec3(const std::string &name, const glm::vec3 &value) const {
        glUniform3fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
    }
};

// type objects
enum class ShapeType { CUBE, SPHERE };

// AABB for bounding limits
struct AABB {
    glm::vec3 min;
    glm::vec3 max;
    bool intersects(const glm::vec3& objMin, const glm::vec3& objMax) const {
        return (min.x <= objMax.x && max.x >= objMin.x) &&
               (min.y <= objMax.y && max.y >= objMin.y) &&
               (min.z <= objMax.z && max.z >= objMin.z);
    }

    bool intersectsSphere(const glm::vec3& sphereCenter, float sphereRadius) const {
        float x = std::max(min.x, std::min(sphereCenter.x, max.x));
        float y = std::max(min.y, std::min(sphereCenter.y, max.y));
        float z = std::max(min.z, std::min(sphereCenter.z, max.z));

        // Calculate distance
        float distance = glm::distance(glm::vec3(x, y, z), sphereCenter);
        return distance < sphereRadius;
    }
};

// General class for cube and sphere
struct Object {
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 color;
    float scale;
    float mass;
    ShapeType type;
    int id;
    AABB collider;

    Object(glm::vec3 p, glm::vec3 v, float s, glm::vec3 c, ShapeType t, int _id) 
        : position(p), velocity(v), scale(s), color(c), type(t), id(_id) {
        mass = (type == ShapeType::CUBE) ? scale * scale * scale : (4.0f/3.0f) * 3.14159f * pow(scale/2.0f, 3);
        updateAABB();
    }
    void update(float dt) {
        position += velocity * dt;
        updateAABB();
    }
    void updateAABB() {
        float halfSize = scale / 2.0f;
        collider.min = position - glm::vec3(halfSize);
        collider.max = position + glm::vec3(halfSize);
    }
};

std::vector<Object> objects;

// Octree node
struct OctreeNode {
    AABB bounds;
    std::vector<Object*> objects;
    int children[8];
    bool isLeaf;

    OctreeNode() {
        reset();
    }

    // Clean and reuse node
    void reset() {
        objects.clear();
        for(int i=0; i<8; i++) {
            children[i] = -1;
        }
        isLeaf = true;
    }
};

class Octree {
    // Configuration
    static const int MAX_OBJECTS = 32;
    static const int MAX_DEPTH = 6;
    
    std::vector<OctreeNode> pool;
    int poolCount;

public:
    // Allocate before
    Octree(int initialSize = 10000) {
        pool.resize(initialSize); 
        poolCount = 0;
    }
    //Reset pool
    void clear() {
        poolCount = 0;
    }

    //Allocate bounds
    void setBounds(AABB rootBounds) {
        clear();
        allocateNode(rootBounds);
    }

    void insertRoot(Object* obj) {
        insertRecursive(0, obj, 0); 
    }

    //Get branch candidates
    void retrieve(std::vector<Object*>& returnObjects, Object* obj) {
        if (poolCount > 0) {
            retrieveRecursive(0, returnObjects, obj);
        }
    }

    //Draw tree
    void drawDebug(Shader& shader, std::function<void()> drawCubeFunc) {
        if (poolCount > 0) {
            //Start with root
            drawRecursive(0, shader, drawCubeFunc);
        }
    }

private:
    int allocateNode(AABB bounds) {
        //Resize pool
        if (poolCount >= pool.size()) {
            pool.resize(pool.size() * 2);
        }
        
        int idx = poolCount++;
        pool[idx].reset();//Clean to reuse it
        pool[idx].bounds = bounds;
        return idx;
    }

    // Split between 8 like chocolate
    void split(int nodeIdx) {
        OctreeNode& node = pool[nodeIdx];
        glm::vec3 min = node.bounds.min;
        glm::vec3 max = node.bounds.max;
        glm::vec3 mid = (min + max) * 0.5f;

        //Morton-code
        for(int i=0; i<8; i++) {
            glm::vec3 newMin, newMax;
            // Store between 8 points
            newMin.x = (i & 1) ? mid.x : min.x; newMax.x = (i & 1) ? max.x : mid.x;
            newMin.y = (i & 2) ? mid.y : min.y; newMax.y = (i & 2) ? max.y : mid.y;
            newMin.z = (i & 4) ? mid.z : min.z; newMax.z = (i & 4) ? max.z : mid.z;
            
            // Store child
            node.children[i] = allocateNode({newMin, newMax});
        }
        node.isLeaf = false;
    }

    // Check child available to store object
    int getIndex(const AABB& bounds, const glm::vec3& minBox, const glm::vec3& maxBox) {
        glm::vec3 mid = (bounds.min + bounds.max) * 0.5f;
        bool right = minBox.x > mid.x; bool left  = maxBox.x < mid.x;
        bool top   = minBox.y > mid.y; bool bottom= maxBox.y < mid.y;
        bool back  = minBox.z > mid.z; bool front = maxBox.z < mid.z;

        int index = -1;
        if (left) {
            if (bottom) index = front ? 0 : (back ? 4 : -1);
            else if (top) index = front ? 2 : (back ? 6 : -1);
        } else if (right) {
            if (bottom) index = front ? 1 : (back ? 5 : -1);
            else if (top) index = front ? 3 : (back ? 7 : -1);
        }
        return index;
    }

    void insertRecursive(int nodeIdx, Object* obj, int level) {
        //Insert in a child
        if (!pool[nodeIdx].isLeaf) {
            //Check position between bounds
            int index = getIndex(pool[nodeIdx].bounds, obj->collider.min, obj->collider.max);
            if (index != -1) {
                insertRecursive(pool[nodeIdx].children[index], obj, level + 1);
                return;
            }
        }

        pool[nodeIdx].objects.push_back(obj);

        // Split if it's full
        if (pool[nodeIdx].objects.size() > MAX_OBJECTS && level < MAX_DEPTH && pool[nodeIdx].isLeaf) {
            split(nodeIdx);
            
            //store temp the objects and move them to the new nodes
            std::vector<Object*> tempObjs = pool[nodeIdx].objects;
            pool[nodeIdx].objects.clear(); 

            for (Object* movingObj : tempObjs) {
                int index = getIndex(pool[nodeIdx].bounds, movingObj->collider.min, movingObj->collider.max);
                if (index != -1) {
                    insertRecursive(pool[nodeIdx].children[index], movingObj, level + 1);
                } else {
                    pool[nodeIdx].objects.push_back(movingObj);
                }
            }
        }
    }

    void retrieveRecursive(int nodeIdx, std::vector<Object*>& returnObjects, Object* obj) {
        OctreeNode& node = pool[nodeIdx]; 
        
        int index = getIndex(node.bounds, obj->collider.min, obj->collider.max);
        
        if (!node.isLeaf && index != -1) { //get next branch
             retrieveRecursive(node.children[index], returnObjects, obj);
        } else if (!node.isLeaf) {
            // check if intersect child objects
            for(int i=0; i<8; i++) {
                if(pool[node.children[i]].bounds.intersects(obj->collider.min, obj->collider.max)) {
                     retrieveRecursive(node.children[i], returnObjects, obj);
                }
            }
        }

        //store objects to returnobjects
        returnObjects.insert(returnObjects.end(), node.objects.begin(), node.objects.end());
    }

    void drawRecursive(int nodeIdx, Shader& shader, std::function<void()> drawFunc) {
        OctreeNode& node = pool[nodeIdx];

        glm::vec3 size = node.bounds.max - node.bounds.min;
        glm::vec3 center = node.bounds.min + (size * 0.5f);

        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, center);
        model = glm::scale(model, size);
        
        shader.setMat4("model", model);

        if (node.isLeaf)
        drawFunc();//draw cube

        if (!node.isLeaf) {
            for (int i = 0; i < 8; i++) {
                //get netx cchild branch
                if (node.children[i] != -1) {
                    drawRecursive(node.children[i], shader, drawFunc);
                }
            }
        }
    }
};

//octree
Octree collisionTree(20000);

// Collisions functions
bool checkCollisionBoxSphere(Object &box, Object &sphere) {
    return box.collider.intersectsSphere(sphere.position, sphere.scale / 2.0f);
}

bool checkCollision(Object &one, Object &two) {
    if (one.type == ShapeType::CUBE && two.type == ShapeType::CUBE) {
        return one.collider.intersects(two.collider.min, two.collider.max);
    } else if (one.type == ShapeType::SPHERE && two.type == ShapeType::SPHERE) {
        return glm::distance(one.position, two.position) < (one.scale + two.scale)/2.0f;
    } else {
        return (one.type == ShapeType::CUBE) ? checkCollisionBoxSphere(one, two) : checkCollisionBoxSphere(two, one);
    }
}

void resolveCollision(Object &one, Object &two) {
    glm::vec3 normal(0,1,0);
    float penetration = 0.0f;

    // Check overlap between SPHERE-SPHERE
    if (one.type == ShapeType::SPHERE && two.type == ShapeType::SPHERE) {
        glm::vec3 d = one.position - two.position;
        float dist = glm::length(d);
        float rSum = (one.scale + two.scale)/2.0f;
        if (dist >= rSum || dist == 0) {
            return;
        }
        normal = glm::normalize(d);
        penetration = rSum - dist;
    } else if (one.type == ShapeType::CUBE && two.type == ShapeType::CUBE) {
        // Check overlap between CUBE-CUBE
        glm::vec3 n = two.position - one.position;
        float x_overlap = (one.scale+two.scale)/2.0f - std::abs(n.x);
        float y_overlap = (one.scale+two.scale)/2.0f - std::abs(n.y);
        float z_overlap = (one.scale+two.scale)/2.0f - std::abs(n.z);
        if (x_overlap<0 || y_overlap<0 || z_overlap<0) {
            return;
        }
        if (x_overlap < y_overlap && x_overlap < z_overlap) {
            normal = glm::vec3(n.x > 0 ? -1 : 1, 0, 0); penetration = x_overlap;
        } else if (y_overlap < z_overlap) {
            normal = glm::vec3(0, n.y > 0 ? -1 : 1, 0); penetration = y_overlap;
        } else {
            normal = glm::vec3(0, 0, n.z > 0 ? -1 : 1); penetration = z_overlap;
        }
    } else {
        // Check overlap between SPHERE-CUBE
        Object* box = (one.type==ShapeType::CUBE) ? &one : &two;
        Object* sph = (one.type==ShapeType::SPHERE) ? &one : &two;
        glm::vec3 c(
            std::max(box->collider.min.x, std::min(sph->position.x, box->collider.max.x)),
            std::max(box->collider.min.y, std::min(sph->position.y, box->collider.max.y)),
            std::max(box->collider.min.z, std::min(sph->position.z, box->collider.max.z))
        );
        glm::vec3 diff = sph->position - c;
        float len = glm::length(diff);
        float rad = sph->scale / 2.0f;
        if (len >= rad && len != 0) {
            return;
        }
        normal = (len == 0) ? glm::vec3(0,1,0) : glm::normalize(diff);
        penetration = rad - len;
        if (one.type == ShapeType::CUBE) {
            normal = -normal;
        }
    }

    // Update positions
    const float percent = 0.8f; 
    const float slop = 0.01f;
    glm::vec3 correction = normal * std::max(penetration - slop, 0.0f) / (1.0f/one.mass + 1.0f/two.mass) * percent;
    one.position += correction * (1.0f/one.mass);
    two.position -= correction * (1.0f/two.mass);
    one.updateAABB(); 
    two.updateAABB();

    glm::vec3 relVel = one.velocity - two.velocity;
    float velAlongNormal = glm::dot(relVel, normal);
    if (velAlongNormal > 0) {
        return;
    }
    float j = -(1.0f + 0.5f) * velAlongNormal;
    j /= (1.0f/one.mass + 1.0f/two.mass);
    glm::vec3 impulse = j * normal;
    one.velocity += impulse * (1.0f/one.mass);
    two.velocity -= impulse * (1.0f/two.mass);
}

void updatePhysics(float dt) {
    if (isPaused) {
        return;
    }

    // Update octree when world change
    if (useOctree) {
        float half = worldSize / 2.0f;
        collisionTree.setBounds(AABB{glm::vec3(-half - 1.0f), glm::vec3(half + 1.0f)});
    }

    float subDt = (dt * timeScale);

    //Update objects positions
    for (auto &obj : objects) {
        obj.update(subDt);
    }

    //Build octree and use it
    if (useOctree) {
        float half = worldSize / 2.0f;
        collisionTree.setBounds(AABB{glm::vec3(-half - 1.0f), glm::vec3(half + 1.0f)});
            
        for (auto &obj : objects) {
            collisionTree.insertRoot(&obj); //Insert objects
        }

        std::vector<Object*> candidates;
        candidates.reserve(objects.size());
        for (size_t i = 0; i < objects.size(); ++i) {
            candidates.clear();
            collisionTree.retrieve(candidates, &objects[i]);//Retrieve branch
            for (auto* other : candidates) {
                if (objects[i].id >= other->id) {
                    continue;
                }
                if (checkCollision(objects[i], *other)) {
                    countCollisions++;
                    resolveCollision(objects[i], *other);
                }
            }
        }
    } else { //Brute force, compare against everyone
        for (size_t i = 0; i < objects.size(); ++i) {
            for (size_t j = i + 1; j < objects.size(); ++j) {
                if (checkCollision(objects[i], objects[j])) {
                    countCollisions++;
                    resolveCollision(objects[i], objects[j]);
                }
            }
        }
    }

    //Check objects against walls
    float bound = worldSize / 2.0f;
    for (auto &obj : objects) {
        bool hit = false;
        float h = obj.scale / 2.0f;
        if (obj.position.x + h > bound)  { obj.position.x = bound - h;  obj.velocity.x *= -1; hit=true; }
        if (obj.position.x - h < -bound) { obj.position.x = -bound + h; obj.velocity.x *= -1; hit=true; }
        if (obj.position.y + h > bound)  { obj.position.y = bound - h;  obj.velocity.y *= -1; hit=true; }
        if (obj.position.y - h < -bound) { obj.position.y = -bound + h; obj.velocity.y *= -1; hit=true; }
        if (obj.position.z + h > bound)  { obj.position.z = bound - h;  obj.velocity.z *= -1; hit=true; }
        if (obj.position.z - h < -bound) { obj.position.z = -bound + h; obj.velocity.z *= -1; hit=true; }
        if (hit) {
            obj.updateAABB();
        }
    }
}

// VAOs
unsigned int cubeVAO = 0, sphereVAO = 0;
unsigned int sphereIndexCount = 0;

void renderCube() {
    if (cubeVAO == 0) {
        //setup draw cube
        float vertices[] = {
            // Back
            -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
             0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
             0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,         
             0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
            -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
            -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
            // Front
            -0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f,
             0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f,
             0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f,
             0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f,
            -0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f,
            -0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f,
            // Left
            -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
            -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
            -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
            -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
            -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
            -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
            // Right
             0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
             0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
             0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
             0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
             0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
             0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
            // Bottom
            -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
             0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
             0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
             0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
            -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
            -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
            // Top
            -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,
             0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
             0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,
             0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
            -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
            -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f
        };
        unsigned int vbo;
        glGenVertexArrays(1, &cubeVAO);
        glGenBuffers(1, &vbo);
        glBindVertexArray(cubeVAO);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float), (void*)(3*sizeof(float)));
    }
    //draw cube
    glBindVertexArray(cubeVAO);
    glDrawArrays(GL_TRIANGLES, 0, 36);
}

void renderSphere() {
    if (sphereVAO == 0) {
        //setup how draw spheres
        glGenVertexArrays(1, &sphereVAO);
        unsigned int vbo, ebo;
        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ebo);
        std::vector<glm::vec3> p, n;
        std::vector<unsigned int> idx;
        const int X_SEG = 32, Y_SEG = 32;
        const float PI = 3.14159265359f;
        //Create vertexes
        for (int x=0; x<=X_SEG; ++x) {
            for (int y=0; y<=Y_SEG; ++y) {
                float xSeg = (float)x/X_SEG, ySeg = (float)y/Y_SEG;
                float xPos = 0.5f * std::cos(xSeg*2.0f*PI) * std::sin(ySeg*PI);
                float yPos = 0.5f * std::cos(ySeg*PI);
                float zPos = 0.5f * std::sin(xSeg*2.0f*PI) * std::sin(ySeg*PI);
                p.push_back(glm::vec3(xPos, yPos, zPos));
                n.push_back(glm::vec3(xPos, yPos, zPos));
            }
        }
        bool oddRow = false;
        //Create mesh
        for (int y=0; y<Y_SEG; ++y) {
            if (!oddRow) {
                for (int x=0; x<=X_SEG; ++x) { 
                    idx.push_back(y*(X_SEG+1)+x); 
                    idx.push_back((y+1)*(X_SEG+1)+x); 
                }
            } else {
                for (int x=X_SEG; x>=0; --x) {
                    idx.push_back((y+1)*(X_SEG+1)+x); 
                    idx.push_back(y*(X_SEG+1)+x);
                }

            }
            oddRow = !oddRow;
        }
        sphereIndexCount = idx.size();
        std::vector<float> data;
        //pack data
        for(size_t i=0; i<p.size(); ++i) {
            data.push_back(p[i].x);
            data.push_back(p[i].y);
            data.push_back(p[i].z);
            data.push_back(n[i].x); 
            data.push_back(n[i].y); 
            data.push_back(n[i].z);
        }
        glBindVertexArray(sphereVAO); glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, data.size()*sizeof(float), &data[0], GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size()*sizeof(unsigned int), &idx[0], GL_STATIC_DRAW);
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float), (void*)0);
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float), (void*)(3*sizeof(float)));
    }
    //Draw spheres
    glBindVertexArray(sphereVAO);
    glDrawElements(GL_TRIANGLE_STRIP, sphereIndexCount, GL_UNSIGNED_INT, 0);
}

void createObject(ShapeType type) {
    float size = 1.0f;
    if (sizeObject == 0) {
        size = randomFloat(0.5f, 0.8f);
    } else if (sizeObject == 1) { 
        size = randomFloat(1.0f, 1.5f);
    } else { //sizeObject = 2
        size = randomFloat(2.0f, 3.0f);
    }

    float limit = worldSize / 2.0f - size;
    if (limit < 0) {
        limit = 0.1f;
    }

    //add object to the list
    objects.emplace_back(
        glm::vec3(randomFloat(-limit, limit), randomFloat(-limit, limit), randomFloat(-limit, limit)),
        glm::vec3(randomFloat(-5, 5), randomFloat(-5, 5), randomFloat(-5, 5)),
        size,
        glm::vec3(randomFloat(0, 1), randomFloat(0, 1), randomFloat(0, 1)),
        type,
        (int)objects.size()
    );
}

//Setup camera
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn) {
    if (isMenuVisible) {
        return;
    }

    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse) { 
        lastX = xpos; 
        lastY = ypos; 
        firstMouse = false; 
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; 
    lastX = xpos;
    lastY = ypos;

    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw += xoffset;
    pitch += yoffset;
    //Ensure pitch stable
    if (pitch > 89.0f) { 
        pitch = 89.0f;
    }
    if (pitch < -89.0f) {
        pitch = -89.0f;
    }

    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(front);
}

//Setup ESC and WSAD key camera movements
void processInput(GLFWwindow *window) {
    if (isMenuVisible) {
        return;
    }

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
    
    float cameraSpeed = 15.0f * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) cameraPos += cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) cameraPos -= cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
}

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);


    GLFWwindow* window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Collisions simulator", NULL, NULL);
    if (!window) { 
        glfwTerminate();
        return -1; 
    }
    glfwMakeContextCurrent(window);

#ifndef __APPLE__
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        return -1;
    }
#endif // !(__APPLE__)

    // Init with menu
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL); 
    glfwSetCursorPosCallback(window, mouse_callback);

    //Use zbuffer
    glEnable(GL_DEPTH_TEST);
    Shader shader(vertexShaderSource, fragmentShaderSource);

    // setup imgui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 410");
    int maxCollitions = 0;

    while (!glfwWindowShouldClose(window)) {
        countCollisions = 0;
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        //Enable/Disable menu
        if (glfwGetKey(window, GLFW_KEY_TAB) == GLFW_PRESS) {
            isMenuVisible = !isMenuVisible;
            if (isMenuVisible) {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            } else {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                firstMouse = true;
            }
            //Keep proccesing in menu
            while(glfwGetKey(window, GLFW_KEY_TAB) == GLFW_PRESS) {
                glfwPollEvents();
            }
        }

        processInput(window);
        updatePhysics(deltaTime);
        maxCollitions = maxCollitions > countCollisions ? maxCollitions : countCollisions;

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shader.use();
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)SCREEN_WIDTH/SCREEN_HEIGHT, 0.1f, 100.0f);
        shader.setMat4("view", view);
        shader.setMat4("projection", projection);
        shader.setVec3("lightPos", glm::vec3(0, 20, 0));
        shader.setVec3("viewPos", cameraPos);

        // Draw world
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::scale(model, glm::vec3(worldSize));
        shader.setMat4("model", model);
        shader.setVec3("objectColor", glm::vec3(1));
        renderCube();
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        // Deaw objects
        for (const auto& obj : objects) {
            model = glm::mat4(1.0f);
            model = glm::translate(model, obj.position);
            model = glm::scale(model, glm::vec3(obj.scale));
            shader.setMat4("model", model);
            shader.setVec3("objectColor", obj.color);
            if (obj.type == ShapeType::CUBE) {
                renderCube();
             } else {
                renderSphere();
             }
        }

        //Debug octotree
        if (useOctree && showOctreeDebug) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            shader.setVec3("objectColor", glm::vec3(0.0f, 1.0f, 0.0f)); //Green color
            collisionTree.drawDebug(shader, renderCube);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }

        // Imgui
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (isMenuVisible) {
            float width = 300.0f;
            ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - width, 0));
            ImGui::SetNextWindowSize(ImVec2(width, io.DisplaySize.y));
            ImGui::Begin("Control", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
            
            ImGui::Text("FPS: %.1f", io.Framerate);
            ImGui::Text("Objects: %d", (int)objects.size());
            ImGui::Text("Max collisions: %d", maxCollitions);
            ImGui::Separator();
            
            ImGui::TextColored(ImVec4(0,1,1,1), "Simulation");
            if (ImGui::Button(isPaused ? "Start" : "Stop", ImVec2(100, 30))) {
                isPaused = !isPaused;
            }
            ImGui::SameLine();
            if (ImGui::Button("Clear all", ImVec2(100, 30))) { 
                objects.clear();
                collisionTree.clear();
                maxCollitions = 0;
            }
            
            ImGui::SliderFloat("Velocity", &timeScale, 0.0f, 10.0f);
            ImGui::SliderFloat("World Size", &worldSize, 5.0f, 50.0f);
            
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0,1,1,1), "Objects");
            ImGui::RadioButton("Small", &sizeObject, 0); ImGui::SameLine();
            ImGui::RadioButton("Medium", &sizeObject, 1); ImGui::SameLine();
            ImGui::RadioButton("Big", &sizeObject, 2);
            
            if (ImGui::Button("Cube", ImVec2(120, 30))) {
                createObject(ShapeType::CUBE);
            }
            ImGui::SameLine();
            if (ImGui::Button("Sphere", ImVec2(120, 30))) {
                createObject(ShapeType::SPHERE);
            }
            if (ImGui::Button("50 Random", ImVec2(120, 30))) { 
                for(int i=0; i<50; i++) {
                    createObject(rand() % 2 ? ShapeType::CUBE : ShapeType::SPHERE); 
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("1000 Random", ImVec2(120, 30))) { 
                for(int i=0; i<1000; i++) {
                    createObject(rand() % 2 ? ShapeType::CUBE : ShapeType::SPHERE); 
                }
            }

            ImGui::Separator();
            ImGui::TextColored(ImVec4(1,1,0,1), "Algorithm");
            ImGui::Checkbox("Octree", &useOctree);
            if (useOctree) {
                ImGui::Text("Using octree");
            }
            else {
                ImGui::Text("Using brute force");
            }

            ImGui::End();
        }

        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            glfwSetWindowShouldClose(window, true);
        }

        //Draw imgui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        //Draw opengl
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    //Clean imgui and glfw
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    return 0;
}