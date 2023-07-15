package controllers

import (
	"context"
	"crypto/md5"
	"encoding/hex"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"math/rand"
	"os"
	"path/filepath"
)

func findDifference(A []string, B []string) []string {
	// Create a map to store strings in B
	bMap := make(map[string]bool)
	for _, b := range B {
		bMap[b] = true
	}

	// Create a slice to store strings in A that are not in B
	var result []string
	for _, a := range A {
		if _, exists := bMap[a]; !exists {
			result = append(result, a)
		}
	}
	return result
}

func contains(s []string, str string) bool {
	for _, v := range s {
		if v == str {
			return true
		}
	}
	return false
}

func getDirectorySize(path string) (float64, error) {
	var size float64 = 0.0
	err := filepath.Walk(path, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			size += float64(info.Size())
		}
		return nil
	})
	return size, err
}

func randomAlphabetic(length int) string {
	const letters = "abcdefghijklmnopqrstuvwxyz"

	b := make([]byte, length)
	for i := range b {
		b[i] = letters[rand.Intn(len(letters))]
	}
	return string(b)
}

func getMongoClient() (*mongo.Client, error) {
	// Set client options
	clientOptions := options.Client().ApplyURI("mongodb://mongo:27017")

	// Connect to MongoDB
	mongoClient, err := mongo.Connect(context.Background(), clientOptions)
	if err != nil {
		return nil, err
	}

	// Check the connection
	err = mongoClient.Ping(context.Background(), nil)
	if err != nil {
		return nil, err
	}

	return mongoClient, nil
}

func sumMap(myMap map[string]float64) float64 {
	var sum = 0.0
	for key := range myMap {
		sum += myMap[key]
	}
	return sum
}

//func getFreeStorageSpace(path string) float64 {
//	stat, err := os.Stat(path)
//	if err != nil {
//		return 0
//	}
//	space := float64(stat.Sys().(*syscall.Statfs_t).Bfree * uint64(stat.Sys().(*syscall.Statfs_t).Bsize))
//	return space
//}

func getHash(s string) string {
	hash := md5.Sum([]byte(s))
	return hex.EncodeToString(hash[:])
}

func hashToInt(h string) int {
	b := []byte(h)
	n := len(b)
	var x int
	for i := 0; i < n; i++ {
		x = (x << 5) + x + int(b[i])
	}
	return x
}
