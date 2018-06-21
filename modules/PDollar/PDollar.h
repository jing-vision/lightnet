#pragma once

#include <string>
#include <vector>

class Point
{
  public:
	Point(double x, double y, int strokeid) : m_X(x), m_Y(y), m_StrokeID(strokeid)
	{
	}

	double getX() const
	{
		return m_X;
	}

	double getY() const
	{
		return m_Y;
	}

	int getStrokeID() const
	{
		return m_StrokeID;
	}

  private:
	double m_X = 0;
	double m_Y = 0;
	int m_StrokeID = 0;
};

class Gesture
{
public:
	
	Gesture(std::vector<Point *> point, std::string gestureName = "");
	~Gesture();

	std::vector<Point *> m_Points;
	std::string m_Name;

	std::vector<Point *> resamplePoints(std::vector<Point *> point, int n);

private:
	std::vector<Point *> scale(std::vector<Point *>point);
	std::vector<Point *> translateTo(std::vector<Point *> point, Point* centroid);
	Point* getCentroid(std::vector<Point *> point);
	double pathLength(std::vector<Point *> point);
};

Gesture *createGesture(std::vector<Point *> point, std::string gestureName);
std::string classify(Gesture *gesture, std::vector<Gesture *> gestureSet);
double cloudMatch(std::vector<Point *> points1, std::vector<Point *> points2);
double cloudDistance(std::vector<Point *> points1, std::vector<Point *> points2, int index);
