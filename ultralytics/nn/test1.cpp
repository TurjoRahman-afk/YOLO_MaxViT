#include <iostream>
using namespace std;

class Student {
    string name;
    int id;
public:
    void input() {
        cin >> name >> id;
    }
    void show() {
        cout << "Name: " << name << ", ID: " << id << endl;
    }
};

int main() {
    Student s[3];
    cout << "Enter 3 students (name id):\n";
    for(int i=0; i<3; i++) s[i].input();

    cout << "\nStudent List:\n";
    for(int i=0; i<3; i++) s[i].show();
    return 0;
}ali